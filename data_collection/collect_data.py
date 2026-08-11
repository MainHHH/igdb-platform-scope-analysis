import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

import requests
from dotenv import load_dotenv


load_dotenv()

CLIENT_ID = os.getenv("IGDB_CLIENT_ID")
CLIENT_SECRET = os.getenv("IGDB_CLIENT_SECRET")

TOKEN_URL = "https://id.twitch.tv/oauth2/token"
GAMES_URL = "https://api.igdb.com/v4/games"

START_TIMESTAMP = int(
    datetime(2017, 1, 1, tzinfo=timezone.utc).timestamp()
)

END_TIMESTAMP = int(
    datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
)

PAGE_SIZE = 500

OUTPUT_DIRECTORY = Path("data_collection/raw_data")
JSON_PATH = OUTPUT_DIRECTORY / "games.json"
CSV_PATH = OUTPUT_DIRECTORY / "games.csv"

FIELDS = [
    # 게임 식별 및 분석 대상 필터
    "id",
    "name",
    "game_type",
    "parent_game",
    "version_parent",
    "game_status",

    # 출시일 및 플랫폼 분류
    "first_release_date",
    "release_dates.date",
    "release_dates.date_format.format",
    "release_dates.platform",
    "platforms.name",

    # 장르 및 평가
    "genres.name",
    "rating",
    "rating_count",
    "aggregated_rating",
    "aggregated_rating_count",
]


def get_access_token():
    """Twitch 인증 서버에서 IGDB API 액세스 토큰을 발급받습니다."""
    token_response = requests.post(
        TOKEN_URL,
        params={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "client_credentials",
        },
        timeout=30,
    )

    token_response.raise_for_status()

    access_token = token_response.json()["access_token"]
    print("IGDB 액세스 토큰 발급 완료")

    return access_token


def create_headers(access_token):
    """IGDB API 요청에 사용할 헤더를 만듭니다."""
    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }

    return headers


def collect_games(headers):
    """조회 기간에 해당하는 게임을 페이지 단위로 수집합니다."""
    all_games = []
    last_game_id = 0

    while True:
        query = f"""
            fields {",".join(FIELDS)};
            where first_release_date >= {START_TIMESTAMP}
                & first_release_date < {END_TIMESTAMP}
                & id > {last_game_id};
            sort id asc;
            limit {PAGE_SIZE};
        """

        response = requests.post(
            GAMES_URL,
            headers=headers,
            data=query,
            timeout=60,
        )

        if response.status_code == 429:
            print("API 요청 한도에 도달했습니다. 2초 후 재시도합니다.")
            time.sleep(2)
            continue

        response.raise_for_status()

        games = response.json()

        if not games:
            break

        all_games.extend(games)
        last_game_id = max(game["id"] for game in games)

        print(
            f"현재 {len(games):,}개 수집, "
            f"누적 {len(all_games):,}개"
        )

        if len(games) < PAGE_SIZE:
            break

        # IGDB 호출 제한을 넘지 않도록 대기
        time.sleep(0.3)

    print(f"총 수집 결과: {len(all_games):,}개")

    return all_games


def save_json(games, output_path):
    """수집한 게임 목록을 JSON 파일로 저장합니다."""
    with output_path.open(
        mode="w",
        encoding="utf-8-sig",
    ) as file:
        json.dump(
            games,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"JSON 저장 완료: {output_path.resolve()}")


def save_csv(games, output_path):
    """수집한 게임 목록을 CSV 파일로 저장합니다."""
    games_df = pd.DataFrame(games)

    games_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"CSV 저장 완료: {output_path.resolve()}")
    print(f"데이터 크기: {games_df.shape}")


def main():
    """IGDB 게임을 수집하고 JSON과 CSV 파일로 저장합니다."""
    if not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError(
            ".env에 IGDB_CLIENT_ID와 "
            "IGDB_CLIENT_SECRET을 설정해주세요."
        )

    access_token = get_access_token()
    headers = create_headers(access_token)
    games = collect_games(headers)

    save_json(games, JSON_PATH)
    save_csv(games, CSV_PATH)


if __name__ == "__main__":
    main()
