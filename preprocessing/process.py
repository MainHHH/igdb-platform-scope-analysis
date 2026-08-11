"""수업에서 배운 문법을 사용한 IGDB 게임 데이터 전처리 코드"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# 경로와 기간 설정
# 파일명은 OUTPUT_FILE_NAME 값만 바꾸면 쉽게 변경할 수 있습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data_collection/raw_data/games.json"
OUTPUT_FOLDER = PROJECT_ROOT / "preprocessing/processed_data"
OUTPUT_FILE_NAME = "games_processed.csv"
OUTPUT_FILE = OUTPUT_FOLDER / OUTPUT_FILE_NAME

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

FINAL_COLUMNS = [
    "id",
    "name",
    "genres",
    "rating",
    "rating_count",
    "release_year",
    "initial_platform_count_30d",
    "initial_platform_group",
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


def make_release_dataframe(games_df):
    """중첩된 출시 기록을 게임-플랫폼-출시일 단위로 변환합니다."""
    games_df = games_df.copy()

    if "release_dates" not in games_df.columns:
        games_df["release_dates"] = None

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


def classify_platforms(games_df, release_df):
    """최초 출시 후 30일 이내 플랫폼 수와 그룹을 계산합니다."""
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

    initial_platforms = make_platform_sets(initial_release_df)

    classification_rows = []

    for game_id in games_df["id"]:
        initial_ids = initial_platforms.get(game_id, set())
        initial_count = len(initial_ids)

        if initial_count == 1:
            platform_group = "single"
        elif initial_count >= 2:
            platform_group = "multi"
        else:
            platform_group = None

        classification_rows.append({
            "id": game_id,
            "initial_platform_count_30d": initial_count,
            "initial_platform_group": platform_group,
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


def make_final_dataframe(games_df, classification_df):
    """게임 정보와 플랫폼 분류를 결합하고 분석용 컬럼을 생성합니다."""
    final_df = pd.merge(
        games_df,
        classification_df,
        on="id",
        how="left",
    )

    final_df = final_df[
        final_df["initial_platform_count_30d"] > 0
    ].copy()

    if "genres" not in final_df.columns:
        final_df["genres"] = None

    final_df["genres"] = final_df["genres"].apply(genre_to_text)
    final_df = final_df[
        final_df["genres"] != ""
    ].copy()

    final_df["rating_count"] = final_df["rating_count"].astype("Int64")

    final_df["release_year"] = final_df[
        "first_release_datetime"
    ].dt.year
    final_df["rating_count_log"] = np.log1p(
        final_df["rating_count"]
    )

    final_df = final_df[
        FINAL_COLUMNS
    ].sort_values(
        by=["release_year", "id"]
    ).reset_index(drop=True)

    if final_df["id"].duplicated().sum() > 0:
        raise ValueError("최종 데이터에 중복 게임 ID가 있습니다.")

    if final_df[["rating", "rating_count"]].isna().sum().sum() > 0:
        raise ValueError("최종 데이터에 평가 결측치가 있습니다.")

    if (final_df["rating_count"] <= 0).sum() > 0:
        raise ValueError("평가 수가 0 이하인 게임이 있습니다.")

    single_error = (
        (final_df["initial_platform_count_30d"] == 1)
        & (final_df["initial_platform_group"] != "single")
    ).sum()
    multi_error = (
        (final_df["initial_platform_count_30d"] >= 2)
        & (final_df["initial_platform_group"] != "multi")
    ).sum()

    if single_error > 0 or multi_error > 0:
        raise ValueError("초기 플랫폼 그룹 분류가 올바르지 않습니다.")

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
    release_df = make_release_dataframe(filtered_df)
    classification_df = classify_platforms(
        filtered_df,
        release_df,
    )
    final_df = make_final_dataframe(
        filtered_df,
        classification_df,
    )
    save_csv(final_df, OUTPUT_FILE)


if __name__ == "__main__":
    main()
