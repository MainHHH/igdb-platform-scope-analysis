"""수업에서 배운 문법을 사용한 IGDB 게임 데이터 전처리 코드"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# 경로와 날짜 설정
# 파일명은 OUTPUT_FILE_NAME 값만 바꾸면 쉽게 변경할 수 있습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data_collection/raw_data/games.json"
OUTPUT_FOLDER = PROJECT_ROOT / "preprocessing/processed_data"
OUTPUT_FILE_NAME = "games_processed.csv"
OUTPUT_FILE = OUTPUT_FOLDER / OUTPUT_FILE_NAME

SNAPSHOT_DATE = "2026-08-10"
START_DATE = "2017-01-01"
END_DATE = "2026-01-01"

EXCLUDED_GAME_STATUS = [2, 3, 4, 6, 7]
EXACT_DATE_FORMAT = "YYYYMMDD"

FILTER_NUMBER_COLUMNS = [
    "game_type",
    "parent_game",
    "version_parent",
    "game_status",
    "rating",
    "rating_count",
]

SOURCE_COLUMNS = [
    "id",
    "name",
    "first_release_date",
    "genres",
    "rating",
    "rating_count",
    "aggregated_rating",
    "aggregated_rating_count",
]

DERIVED_COLUMNS = [
    "release_year",
    "release_age_days",
    "initial_platform_count_30d",
    "initial_platform_group",
    "lifetime_platform_count",
    "expansion_status",
    "rating_count_log",
]


def load_json(file_path):
    """JSON 파일을 읽어서 데이터프레임으로 변환합니다."""
    with open(file_path, "r", encoding="utf-8-sig") as file:
        games = json.load(file)

    games_df = pd.DataFrame(games)
    print("원본 게임 수:", len(games_df))
    return games_df


def filter_games(games_df):
    """기간, 본편, 에디션, 상태, 평가 조건으로 게임을 필터링합니다."""
    games_df = games_df.copy()

    # IGDB는 값이 없으면 필드를 생략할 수 있으므로 없는 컬럼을 추가합니다.
    for column in FILTER_NUMBER_COLUMNS:
        if column not in games_df.columns:
            games_df[column] = None

    games_df[FILTER_NUMBER_COLUMNS] = games_df[
        FILTER_NUMBER_COLUMNS
    ].apply(pd.to_numeric, errors="coerce")

    games_df["first_release_datetime"] = pd.to_datetime(
        games_df["first_release_date"],
        unit="s",
        utc=True,
        errors="coerce",
    )

    start_date = pd.to_datetime(START_DATE, utc=True)
    end_date = pd.to_datetime(END_DATE, utc=True)

    period_condition = (
        (games_df["first_release_datetime"] >= start_date)
        & (games_df["first_release_datetime"] < end_date)
    )
    main_game_condition = games_df["game_type"] == 0
    parent_condition = games_df["parent_game"].isna()
    version_condition = games_df["version_parent"].isna()
    status_condition = ~games_df["game_status"].isin(EXCLUDED_GAME_STATUS)
    rating_condition = (
        games_df["rating"].notna()
        & (games_df["rating_count"] > 0)
    )

    filtered_df = games_df[
        period_condition
        & main_game_condition
        & parent_condition
        & version_condition
        & status_condition
        & rating_condition
    ].copy()

    print("기본 필터 후 게임 수:", len(filtered_df))
    return filtered_df


def make_platform_dataframe(games_df):
    """게임별 플랫폼 목록을 게임-플랫폼 단위 행으로 변환합니다."""
    platform_rows = []

    for game_id, platforms in zip(games_df["id"], games_df["platforms"]):
        if isinstance(platforms, list):
            for platform in platforms:
                if isinstance(platform, dict):
                    platform_id = platform.get("id")
                    platform_name = platform.get("name")

                    if platform_id is not None:
                        platform_rows.append({
                            "id": game_id,
                            "platform_id": platform_id,
                            "platform_name": platform_name,
                        })

    platform_df = pd.DataFrame(
        platform_rows,
        columns=["id", "platform_id", "platform_name"],
    )

    platform_df["platform_id"] = pd.to_numeric(
        platform_df["platform_id"],
        errors="coerce",
    )

    duplicate_count = platform_df.duplicated(
        subset=["id", "platform_id"]
    ).sum()

    platform_df = platform_df.dropna(subset=["platform_id"])
    platform_df = platform_df.drop_duplicates(subset=["id", "platform_id"])

    print("플랫폼 중복 기록 수:", duplicate_count)
    return platform_df


def make_release_dataframe(games_df):
    """중첩된 출시 기록을 게임-플랫폼-출시일 단위로 변환합니다."""
    release_rows = []

    for game_id, release_dates in zip(games_df["id"], games_df["release_dates"]):
        if isinstance(release_dates, list):
            for release in release_dates:
                if not isinstance(release, dict):
                    continue

                platform = release.get("platform")
                date_format = release.get("date_format")

                if isinstance(platform, dict):
                    platform_id = platform.get("id")
                else:
                    platform_id = platform

                if isinstance(date_format, dict):
                    date_format_name = date_format.get("format")
                else:
                    date_format_name = None

                release_rows.append({
                    "id": game_id,
                    "platform_id": platform_id,
                    "release_timestamp": release.get("date"),
                    "date_format": date_format_name,
                })

    release_df = pd.DataFrame(
        release_rows,
        columns=[
            "id",
            "platform_id",
            "release_timestamp",
            "date_format",
        ],
    )

    number_columns = ["platform_id", "release_timestamp"]
    release_df[number_columns] = release_df[number_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    release_df["release_datetime"] = pd.to_datetime(
        release_df["release_timestamp"],
        unit="s",
        utc=True,
        errors="coerce",
    )

    print("전체 출시 기록 수:", len(release_df))
    return release_df


def make_platform_sets(dataframe):
    """게임 ID별 플랫폼 ID를 집합으로 묶습니다."""
    platform_sets = {}

    for game_id, group_df in dataframe.groupby("id"):
        platform_ids = set(group_df["platform_id"].astype(int))
        platform_sets[game_id] = platform_ids

    return platform_sets


def classify_platforms(games_df, platform_df, release_df):
    """초기 플랫폼 수와 30일 이후 확장 상태를 계산합니다."""
    exact_release_df = release_df[
        (release_df["date_format"] == EXACT_DATE_FORMAT)
        & release_df["platform_id"].notna()
        & release_df["release_datetime"].notna()
    ].copy()

    exact_release_df = pd.merge(
        exact_release_df,
        games_df[["id", "first_release_datetime"]],
        on="id",
        how="left",
    )

    exact_release_df["days_from_first"] = (
        exact_release_df["release_datetime"]
        - exact_release_df["first_release_datetime"]
    ).dt.days

    initial_release_df = exact_release_df[
        (exact_release_df["days_from_first"] >= 0)
        & (exact_release_df["days_from_first"] <= 30)
    ].copy()

    initial_release_df = initial_release_df.sort_values(
        "release_datetime"
    ).drop_duplicates(
        subset=["id", "platform_id"],
        keep="first",
    )

    after_30d_release_df = exact_release_df[
        exact_release_df["days_from_first"] > 30
    ].drop_duplicates(
        subset=["id", "platform_id"]
    )

    initial_platforms = make_platform_sets(initial_release_df)
    lifetime_platforms = make_platform_sets(platform_df)
    after_30d_platforms = make_platform_sets(after_30d_release_df)

    classification_rows = []

    for game_id in games_df["id"]:
        initial_ids = initial_platforms.get(game_id, set())
        lifetime_ids = lifetime_platforms.get(game_id, set())
        after_30d_ids = after_30d_platforms.get(game_id, set())

        initial_count = len(initial_ids)
        lifetime_count = len(lifetime_ids)

        if initial_count == 1:
            platform_group = "single"
        elif initial_count >= 2:
            platform_group = "multi"
        else:
            platform_group = None

        non_initial_ids = lifetime_ids - initial_ids
        confirmed_ids = (after_30d_ids - initial_ids) & lifetime_ids

        if len(confirmed_ids) > 0:
            expansion_status = "confirmed_expanded"
        elif len(non_initial_ids) == 0:
            expansion_status = "not_expanded"
        else:
            expansion_status = "unknown"

        if initial_count > lifetime_count:
            raise ValueError(
                "초기 플랫폼 수가 전체 플랫폼 수보다 큽니다."
            )

        classification_rows.append({
            "id": game_id,
            "initial_platform_count_30d": initial_count,
            "initial_platform_group": platform_group,
            "lifetime_platform_count": lifetime_count,
            "expansion_status": expansion_status,
        })

    classification_df = pd.DataFrame(classification_rows)

    unclassified_count = (
        classification_df["initial_platform_count_30d"] == 0
    ).sum()

    print("플랫폼 분류 불가능 게임 수:", unclassified_count)
    return classification_df


def genre_to_text(genres):
    """장르 객체 목록을 |로 연결된 문자열로 변환합니다."""
    genre_names = []

    if isinstance(genres, list):
        for genre in genres:
            if isinstance(genre, dict):
                genre_name = genre.get("name")

                if genre_name and genre_name not in genre_names:
                    genre_names.append(genre_name)

    genre_names.sort()
    return "|".join(genre_names)


def make_final_dataframe(games_df, classification_df, snapshot_date):
    """게임 정보와 플랫폼 분류를 결합하고 최종 컬럼을 생성합니다."""
    final_df = pd.merge(
        games_df,
        classification_df,
        on="id",
        how="left",
    )

    final_df = final_df[
        final_df["initial_platform_count_30d"] > 0
    ].copy()

    # IGDB에서 값이 없으면 해당 컬럼 자체가 생략될 수 있습니다.
    for column in SOURCE_COLUMNS:
        if column not in final_df.columns:
            final_df[column] = None

    final_df["genres"] = final_df["genres"].apply(genre_to_text)
    final_df["rating_count"] = final_df["rating_count"].astype("Int64")

    final_df["aggregated_rating_count"] = pd.to_numeric(
        final_df["aggregated_rating_count"],
        errors="coerce",
    ).fillna(0).astype("Int64")

    snapshot_datetime = pd.to_datetime(snapshot_date, utc=True)

    final_df["release_year"] = final_df[
        "first_release_datetime"
    ].dt.year
    final_df["release_age_days"] = (
        snapshot_datetime
        - final_df["first_release_datetime"]
    ).dt.days
    final_df["rating_count_log"] = np.log1p(
        final_df["rating_count"]
    )
    final_df["first_release_date"] = final_df[
        "first_release_datetime"
    ].dt.strftime("%Y-%m-%d")

    final_df = final_df[
        SOURCE_COLUMNS + DERIVED_COLUMNS
    ].sort_values(
        by=["release_year", "id"]
    ).reset_index(drop=True)

    if final_df["id"].duplicated().sum() > 0:
        raise ValueError("최종 데이터에 중복 게임 ID가 있습니다.")

    if (
        final_df["initial_platform_count_30d"]
        > final_df["lifetime_platform_count"]
    ).sum() > 0:
        raise ValueError(
            "초기 플랫폼 수가 전체 플랫폼 수보다 큽니다."
        )

    print("최종 게임 수:", len(final_df))
    return final_df


def save_csv(final_df, output_path):
    """최종 데이터프레임을 CSV 파일로 저장합니다."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("CSV 저장 완료:", output_path)


def main():
    games_df = load_json(INPUT_FILE)
    filtered_df = filter_games(games_df)
    platform_df = make_platform_dataframe(filtered_df)
    release_df = make_release_dataframe(filtered_df)
    classification_df = classify_platforms(
        filtered_df,
        platform_df,
        release_df,
    )
    final_df = make_final_dataframe(
        filtered_df,
        classification_df,
        SNAPSHOT_DATE,
    )
    save_csv(final_df, OUTPUT_FILE)


if __name__ == "__main__":
    main()
