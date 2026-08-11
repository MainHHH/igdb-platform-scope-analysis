from pathlib import Path

import koreanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


DATA_FILE = (
    Path(__file__).resolve().parent
    / "preprocessing"
    / "processed_data"
    / "games_processed.csv"
)

REQUIRED_COLUMNS = [
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

PLATFORM_ORDER = ["single", "multi"]
PLATFORM_LABELS = {"single": "초기 단일", "multi": "초기 멀티"}
PLATFORM_COLORS = {"single": "#4C78A8", "multi": "#F58518"}
BUCKET_ORDER = ["2개", "3개", "4개", "5개 이상"]


@st.cache_data(show_spinner=False)
def load_data(file_path):
    """전처리된 CSV를 읽고 대시보드 필수 컬럼을 확인한다."""
    dataframe = pd.read_csv(file_path)
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in dataframe.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"필수 컬럼이 없습니다: {missing_text}")

    return dataframe


def make_genre_dataframe(dataframe):
    """`|`로 저장된 장르를 게임-장르 단위의 행으로 펼친다."""
    genre_dataframe = dataframe.copy()
    genre_dataframe["genre"] = (
        genre_dataframe["genres"].fillna("").str.split("|")
    )
    genre_dataframe = genre_dataframe.explode("genre")
    genre_dataframe["genre"] = genre_dataframe["genre"].str.strip()
    genre_dataframe = genre_dataframe[genre_dataframe["genre"] != ""]

    return genre_dataframe


def filter_data(dataframe, year_range, selected_genres):
    """출시 연도와 선택 장르 중 하나 이상에 해당하는 게임만 남긴다."""
    start_year, end_year = year_range
    filtered = dataframe[
        dataframe["release_year"].between(start_year, end_year)
    ].copy()

    if selected_genres:
        genre_dataframe = make_genre_dataframe(filtered)
        selected_ids = genre_dataframe[
            genre_dataframe["genre"].isin(selected_genres)
        ]["id"].unique()
        filtered = filtered[filtered["id"].isin(selected_ids)].copy()

    return filtered


def make_group_summary(dataframe):
    """초기 플랫폼 그룹별 게임 수와 평가 지표를 요약한다."""
    summary = (
        dataframe.groupby("initial_platform_group")
        .agg(
            game_count=("id", "nunique"),
            rating_count_median=("rating_count", "median"),
            rating_log_q1=("rating_count_log", lambda values: values.quantile(0.25)),
            rating_log_median=("rating_count_log", "median"),
            rating_log_q3=("rating_count_log", lambda values: values.quantile(0.75)),
            rating_q1=("rating", lambda values: values.quantile(0.25)),
            rating_median=("rating", "median"),
            rating_q3=("rating", lambda values: values.quantile(0.75)),
        )
        .reindex(PLATFORM_ORDER)
        .dropna(subset=["game_count"])
        .reset_index()
    )

    return summary


def make_genre_year_stats(dataframe):
    """장르·연도·플랫폼 그룹별 표본 수와 중앙값을 계산한다."""
    genre_dataframe = make_genre_dataframe(dataframe)
    stats = (
        genre_dataframe.groupby(
            ["genre", "release_year", "initial_platform_group"]
        )
        .agg(
            game_count=("id", "nunique"),
            rating_count_median=("rating_count", "median"),
            rating_log_median=("rating_count_log", "median"),
            rating_median=("rating", "median"),
        )
        .reset_index()
    )

    return stats


def make_genre_year_comparison(dataframe, min_sample):
    """양쪽 집단의 표본 기준을 충족한 장르·연도별 중앙값 차이를 만든다."""
    stats = make_genre_year_stats(dataframe)
    single = stats[stats["initial_platform_group"] == "single"].copy()
    multi = stats[stats["initial_platform_group"] == "multi"].copy()

    single = single.rename(
        columns={
            "game_count": "single_count",
            "rating_count_median": "single_rating_count_median",
            "rating_log_median": "single_rating_log_median",
            "rating_median": "single_rating_median",
        }
    ).drop(columns="initial_platform_group")

    multi = multi.rename(
        columns={
            "game_count": "multi_count",
            "rating_count_median": "multi_rating_count_median",
            "rating_log_median": "multi_rating_log_median",
            "rating_median": "multi_rating_median",
        }
    ).drop(columns="initial_platform_group")

    comparison = single.merge(multi, on=["genre", "release_year"])
    comparison = comparison[
        (comparison["single_count"] >= min_sample)
        & (comparison["multi_count"] >= min_sample)
    ].copy()
    comparison["median_gap"] = (
        comparison["multi_rating_log_median"]
        - comparison["single_rating_log_median"]
    )

    return comparison.sort_values(["genre", "release_year"]).reset_index(drop=True)


def make_platform_bucket_stats(dataframe):
    """초기 멀티플랫폼 게임을 플랫폼 수 구간별로 요약한다."""
    multi = dataframe[
        dataframe["initial_platform_group"] == "multi"
    ].copy()
    multi["platform_bucket"] = pd.cut(
        multi["initial_platform_count_30d"],
        bins=[1, 2, 3, 4, np.inf],
        labels=BUCKET_ORDER,
        include_lowest=True,
        right=True,
    )

    stats = (
        multi.groupby("platform_bucket", observed=False)
        .agg(
            game_count=("id", "nunique"),
            rating_count_median=("rating_count", "median"),
            rating_log_q1=("rating_count_log", lambda values: values.quantile(0.25)),
            rating_log_median=("rating_count_log", "median"),
            rating_log_q3=("rating_count_log", lambda values: values.quantile(0.75)),
        )
        .reset_index()
    )
    stats["platform_bucket"] = stats["platform_bucket"].astype(str)

    correlation = multi[
        ["initial_platform_count_30d", "rating_count_log"]
    ].corr(method="spearman").iloc[0, 1]

    return stats, correlation


def make_representative_games(dataframe):
    """플랫폼 그룹별 상위 3개·중앙값 근처 1개·하위 3개 게임을 고른다."""
    selected_rows = []

    for group in PLATFORM_ORDER:
        group_data = dataframe[
            dataframe["initial_platform_group"] == group
        ].copy()

        if group_data.empty:
            continue

        top = group_data.nlargest(3, "rating_count_log").copy()
        top["case"] = "상위 평가 참여"

        median_value = group_data["rating_count_log"].median()
        group_data["median_distance"] = (
            group_data["rating_count_log"] - median_value
        ).abs()
        middle = group_data.nsmallest(1, "median_distance").copy()
        middle["case"] = "중앙값 근처"

        bottom = group_data.nsmallest(3, "rating_count_log").copy()
        bottom["case"] = "하위 평가 참여"

        selected_rows.extend([top, middle, bottom])

    if not selected_rows:
        return pd.DataFrame()

    representatives = pd.concat(selected_rows, ignore_index=True)
    representatives = representatives.drop_duplicates(subset="id")

    return representatives


def format_number(value):
    """대시보드 숫자를 천 단위 구분 기호와 함께 표시한다."""
    if pd.isna(value):
        return "-"
    return f"{value:,.0f}"


def add_group_labels(dataframe):
    """표에 사용할 한글 플랫폼 그룹명을 추가한다."""
    result = dataframe.copy()
    result["플랫폼 그룹"] = result["initial_platform_group"].map(PLATFORM_LABELS)
    return result


def show_overall_tab(dataframe):
    st.subheader("초기 단일·멀티플랫폼 게임의 평가 참여도와 이용자 평점은 어떻게 다른가?")
    st.caption(
        f"현재 조건: 고유 게임 {dataframe['id'].nunique():,}개 · "
        "평가가 존재하는 게임만 포함"
    )

    summary = make_group_summary(dataframe)
    summary_index = summary.set_index("initial_platform_group")

    if not set(PLATFORM_ORDER).issubset(summary_index.index):
        st.warning("현재 조건에는 초기 단일 또는 초기 멀티 그룹의 데이터가 없습니다.")
        return

    single = summary_index.loc["single"]
    multi = summary_index.loc["multi"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("분석 게임", f"{dataframe['id'].nunique():,}개")
    metric_columns[1].metric(
        "초기 단일 평가 수 중앙값", format_number(single["rating_count_median"])
    )
    metric_columns[2].metric(
        "초기 멀티 평가 수 중앙값", format_number(multi["rating_count_median"])
    )
    metric_columns[3].metric(
        "평점 중앙값 차이",
        f"{multi['rating_median'] - single['rating_median']:+.1f}점",
        help="초기 멀티 중앙값 - 초기 단일 중앙값",
    )

    st.markdown("#### 평가 참여도 분포")
    figure, axis = plt.subplots(figsize=(10, 4.8))
    sns.boxplot(
        data=dataframe,
        x="initial_platform_group",
        y="rating_count_log",
        hue="initial_platform_group",
        order=PLATFORM_ORDER,
        hue_order=PLATFORM_ORDER,
        palette=PLATFORM_COLORS,
        showfliers=False,
        legend=False,
        ax=axis,
    )
    axis.set_xlabel("")
    axis.set_ylabel("평가 수 로그값 log1p(rating_count)")
    axis.set_xticks([0, 1], ["초기 단일", "초기 멀티"])
    axis.set_title("초기 플랫폼 그룹별 평가 참여도 분포")
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    detail_column, table_column = st.columns([1, 1.25])
    with detail_column:
        st.markdown("#### 이용자 평점 중앙값과 IQR")
        rating_figure, rating_axis = plt.subplots(figsize=(6.2, 4.1))
        x_values = np.arange(len(summary))
        lower_error = summary["rating_median"] - summary["rating_q1"]
        upper_error = summary["rating_q3"] - summary["rating_median"]
        colors = [PLATFORM_COLORS[group] for group in summary["initial_platform_group"]]
        rating_axis.errorbar(
            x_values,
            summary["rating_median"],
            yerr=[lower_error, upper_error],
            fmt="none",
            ecolor="#7F7F7F",
            elinewidth=3,
            capsize=7,
        )
        rating_axis.scatter(
            x_values,
            summary["rating_median"],
            c=colors,
            s=90,
            zorder=3,
        )
        rating_axis.set_xticks(x_values, [PLATFORM_LABELS[group] for group in summary["initial_platform_group"]])
        rating_axis.set_ylabel("IGDB 이용자 평점")
        rating_axis.set_ylim(0, 100)
        rating_axis.grid(axis="x", visible=False)
        st.pyplot(rating_figure, width="stretch")
        plt.close(rating_figure)

    with table_column:
        st.markdown("#### 그룹별 요약")
        display_summary = add_group_labels(summary)
        display_summary = display_summary.rename(
            columns={
                "game_count": "게임 수",
                "rating_count_median": "평가 수 중앙값",
                "rating_log_median": "평가 수 로그 중앙값",
                "rating_median": "평점 중앙값",
            }
        )[
            [
                "플랫폼 그룹",
                "게임 수",
                "평가 수 중앙값",
                "평가 수 로그 중앙값",
                "평점 중앙값",
            ]
        ]
        st.dataframe(
            display_summary.style.format(
                {
                    "게임 수": "{:,.0f}",
                    "평가 수 중앙값": "{:,.1f}",
                    "평가 수 로그 중앙값": "{:.2f}",
                    "평점 중앙값": "{:.1f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    if multi["rating_count_median"] > single["rating_count_median"]:
        st.info(
            "현재 조건에서는 초기 멀티플랫폼 게임의 평가 수 중앙값이 더 높게 관찰됩니다. "
            "이는 플랫폼 전략의 인과 효과가 아니라 평가가 존재하는 게임 안에서의 연관성입니다."
        )
    else:
        st.info(
            "현재 조건에서는 초기 단일플랫폼 게임의 평가 수 중앙값이 같거나 더 높게 관찰됩니다. "
            "장르와 출시 연도에 따라 결과가 달라지는지 다음 탭에서 확인할 수 있습니다."
        )


def show_consistency_tab(dataframe, min_sample):
    st.subheader("초기 단일·멀티플랫폼의 차이가 출시 연도와 장르별로도 반복되는가?")
    st.caption(
        f"현재 조건: 장르×출시 연도별로 양쪽 그룹에 각각 최소 {min_sample}개 게임 적용"
    )
    comparison = make_genre_year_comparison(dataframe, min_sample)

    if comparison.empty:
        st.warning("현재 조건에서 양쪽 집단의 최소 표본을 충족하는 장르·연도 조합이 없습니다.")
        return

    positive_count = int((comparison["median_gap"] > 0).sum())
    negative_count = int((comparison["median_gap"] < 0).sum())
    metric_columns = st.columns(4)
    metric_columns[0].metric("유효 장르×연도", f"{len(comparison):,}개")
    metric_columns[1].metric("초기 멀티 우세", f"{positive_count:,}개")
    metric_columns[2].metric("초기 단일 우세", f"{negative_count:,}개")
    metric_columns[3].metric(
        "멀티 우세 비율", f"{positive_count / len(comparison) * 100:.1f}%"
    )

    st.markdown("#### 장르·연도별 평가 참여도 중앙값 차이")
    heatmap_data = comparison.pivot(
        index="genre", columns="release_year", values="median_gap"
    )
    max_absolute = comparison["median_gap"].abs().max()
    figure_height = max(5, len(heatmap_data) * 0.45)
    figure, axis = plt.subplots(figsize=(12, figure_height))
    sns.heatmap(
        heatmap_data,
        cmap="RdBu_r",
        center=0,
        vmin=-max_absolute,
        vmax=max_absolute,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "멀티 중앙값 - 단일 중앙값"},
        ax=axis,
    )
    axis.set_xlabel("출시 연도")
    axis.set_ylabel("장르")
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.caption("양수(붉은색)는 초기 멀티, 음수(푸른색)는 초기 단일의 평가 참여도 중앙값이 더 높음을 뜻합니다.")

    available_genres = sorted(comparison["genre"].unique())
    selected_genre = st.selectbox(
        "세부 추이를 확인할 장르",
        available_genres,
        key="consistency_genre",
    )
    selected_comparison = comparison[
        comparison["genre"] == selected_genre
    ].sort_values("release_year")

    line_data = selected_comparison.melt(
        id_vars="release_year",
        value_vars=["single_rating_log_median", "multi_rating_log_median"],
        var_name="group",
        value_name="rating_log_median",
    )
    line_data["group"] = line_data["group"].map(
        {
            "single_rating_log_median": "single",
            "multi_rating_log_median": "multi",
        }
    )

    line_column, table_column = st.columns([1.2, 1])
    with line_column:
        st.markdown(f"#### {selected_genre}의 실제 중앙값")
        line_figure, line_axis = plt.subplots(figsize=(7, 4.2))
        sns.lineplot(
            data=line_data,
            x="release_year",
            y="rating_log_median",
            hue="group",
            style="group",
            hue_order=PLATFORM_ORDER,
            palette=PLATFORM_COLORS,
            markers=True,
            dashes=False,
            ax=line_axis,
        )
        line_axis.set_xlabel("출시 연도")
        line_axis.set_ylabel("평가 수 로그 중앙값")
        line_axis.legend(title="초기 플랫폼 그룹", labels=["초기 단일", "초기 멀티"])
        st.pyplot(line_figure, width="stretch")
        plt.close(line_figure)

    with table_column:
        st.markdown("#### 선택 장르 상세값")
        display_table = selected_comparison.rename(
            columns={
                "release_year": "출시 연도",
                "single_count": "단일 게임 수",
                "multi_count": "멀티 게임 수",
                "single_rating_log_median": "단일 중앙값",
                "multi_rating_log_median": "멀티 중앙값",
                "median_gap": "중앙값 차이",
            }
        )[
            [
                "출시 연도",
                "단일 게임 수",
                "멀티 게임 수",
                "단일 중앙값",
                "멀티 중앙값",
                "중앙값 차이",
            ]
        ]
        st.dataframe(
            display_table.style.format(
                {
                    "단일 중앙값": "{:.2f}",
                    "멀티 중앙값": "{:.2f}",
                    "중앙값 차이": "{:+.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.info("장르·연도별 차이의 방향과 크기가 얼마나 일관적인지를 확인하는 화면이며, 원인을 증명하지는 않습니다.")


def show_platform_count_tab(dataframe):
    st.subheader("초기 멀티플랫폼 게임은 초기 플랫폼 수가 많을수록 평가 참여도가 높은가?")
    st.caption("현재 조건: 초기 멀티플랫폼 게임만 사용 · 플랫폼 수를 2개, 3개, 4개, 5개 이상으로 구분")
    stats, correlation = make_platform_bucket_stats(dataframe)
    multi_count = dataframe[
        dataframe["initial_platform_group"] == "multi"
    ]["id"].nunique()

    if multi_count == 0:
        st.warning("현재 조건에는 초기 멀티플랫폼 게임이 없습니다.")
        return

    nonempty_stats = stats[stats["game_count"] > 0].copy()
    highest_bucket = "-"
    if not nonempty_stats.empty:
        highest_bucket = nonempty_stats.loc[
            nonempty_stats["rating_log_median"].idxmax(), "platform_bucket"
        ]

    metric_columns = st.columns(3)
    metric_columns[0].metric("초기 멀티 게임", f"{multi_count:,}개")
    metric_columns[1].metric(
        "Spearman ρ", "-" if pd.isna(correlation) else f"{correlation:.3f}"
    )
    metric_columns[2].metric("중앙값 최고 구간", highest_bucket)

    chart_column, table_column = st.columns([1.25, 1])
    with chart_column:
        st.markdown("#### 플랫폼 수 구간별 평가 참여도 중앙값과 IQR")
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        chart_data = stats[stats["game_count"] > 0].copy()
        x_values = np.arange(len(chart_data))
        lower_error = chart_data["rating_log_median"] - chart_data["rating_log_q1"]
        upper_error = chart_data["rating_log_q3"] - chart_data["rating_log_median"]
        axis.errorbar(
            x_values,
            chart_data["rating_log_median"],
            yerr=[lower_error, upper_error],
            fmt="o-",
            color="#6F4E7C",
            linewidth=2,
            markersize=7,
            capsize=6,
        )
        axis.set_xticks(x_values, chart_data["platform_bucket"])
        axis.set_xlabel("출시 후 30일 이내 플랫폼 수")
        axis.set_ylabel("평가 수 로그값 중앙값")
        axis.grid(axis="x", visible=False)
        st.pyplot(figure, width="stretch")
        plt.close(figure)

    with table_column:
        st.markdown("#### 구간별 상세값")
        display_table = stats.rename(
            columns={
                "platform_bucket": "플랫폼 수 구간",
                "game_count": "게임 수",
                "rating_count_median": "실제 평가 수 중앙값",
                "rating_log_q1": "로그 Q1",
                "rating_log_median": "로그 중앙값",
                "rating_log_q3": "로그 Q3",
            }
        )
        st.dataframe(
            display_table.style.format(
                {
                    "게임 수": "{:,.0f}",
                    "실제 평가 수 중앙값": "{:,.1f}",
                    "로그 Q1": "{:.2f}",
                    "로그 중앙값": "{:.2f}",
                    "로그 Q3": "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    if pd.isna(correlation):
        relationship_text = "상관계수를 계산할 수 없습니다."
    elif abs(correlation) < 0.2:
        relationship_text = "매우 약한 연관성"
    elif abs(correlation) < 0.4:
        relationship_text = "약한 연관성"
    elif abs(correlation) < 0.6:
        relationship_text = "보통 수준의 연관성"
    else:
        relationship_text = "비교적 강한 연관성"

    st.info(
        f"초기 플랫폼 수와 평가 참여도 사이에는 {relationship_text}이 관찰됩니다. "
        "상관계수는 플랫폼 수가 관심도를 높였다는 인과관계를 의미하지 않습니다."
    )


def show_exception_tab(dataframe, min_sample):
    st.subheader("초기 멀티보다 초기 단일의 평가 참여도가 높은 예외 조건은 무엇인가?")
    st.caption(
        f"현재 조건: 장르×출시 연도별 양쪽 그룹 최소 {min_sample}개 · "
        "평가 참여도 중앙값 차이가 작은 순서"
    )
    comparison = make_genre_year_comparison(dataframe, min_sample)
    exceptions = comparison.nsmallest(10, "median_gap").copy()

    if exceptions.empty:
        st.warning("현재 조건에서 비교 가능한 장르·연도 조합이 없습니다.")
        return

    negative_count = int((comparison["median_gap"] < 0).sum())
    minimum_row = exceptions.iloc[0]
    metric_columns = st.columns(3)
    metric_columns[0].metric("비교 가능 조합", f"{len(comparison):,}개")
    metric_columns[1].metric("초기 단일 우세 조합", f"{negative_count:,}개")
    metric_columns[2].metric("가장 작은 차이", f"{minimum_row['median_gap']:+.2f}")

    exceptions["condition"] = (
        exceptions["genre"] + " · " + exceptions["release_year"].astype(str)
    )
    st.markdown("#### 평가 참여도 중앙값 차이가 작은 10개 조건")
    figure, axis = plt.subplots(figsize=(10, 5.5))
    ordered = exceptions.sort_values("median_gap")
    colors = [
        PLATFORM_COLORS["single"] if value < 0 else PLATFORM_COLORS["multi"]
        for value in ordered["median_gap"]
    ]
    axis.barh(ordered["condition"], ordered["median_gap"], color=colors)
    axis.axvline(0, color="#444444", linewidth=1)
    axis.set_xlabel("평가 수 로그 중앙값 차이 (멀티 - 단일)")
    axis.set_ylabel("")
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    selected_condition = st.selectbox(
        "상세 비교할 조건",
        exceptions["condition"].tolist(),
        key="exception_condition",
    )
    selected_row = exceptions[
        exceptions["condition"] == selected_condition
    ].iloc[0]

    chart_column, table_column = st.columns([1, 1.3])
    with chart_column:
        st.markdown("#### 선택 조건의 실제 중앙값")
        dumbbell_figure, dumbbell_axis = plt.subplots(figsize=(6.5, 2.8))
        single_value = selected_row["single_rating_count_median"]
        multi_value = selected_row["multi_rating_count_median"]
        dumbbell_axis.plot(
            [single_value, multi_value],
            [0, 0],
            color="#B8B8B8",
            linewidth=4,
            zorder=1,
        )
        dumbbell_axis.scatter(
            [single_value, multi_value],
            [0, 0],
            c=[PLATFORM_COLORS["single"], PLATFORM_COLORS["multi"]],
            s=130,
            zorder=2,
        )
        dumbbell_axis.text(single_value, 0.08, "초기 단일", ha="center")
        dumbbell_axis.text(multi_value, -0.10, "초기 멀티", ha="center")
        dumbbell_axis.set_xlabel("실제 평가 수 중앙값")
        dumbbell_axis.set_yticks([])
        st.pyplot(dumbbell_figure, width="stretch")
        plt.close(dumbbell_figure)

    with table_column:
        st.markdown("#### 예외 조건 상세표")
        display_table = exceptions.rename(
            columns={
                "genre": "장르",
                "release_year": "출시 연도",
                "single_count": "단일 게임 수",
                "multi_count": "멀티 게임 수",
                "single_rating_count_median": "단일 실제 평가 수 중앙값",
                "multi_rating_count_median": "멀티 실제 평가 수 중앙값",
                "median_gap": "로그 중앙값 차이",
            }
        )[
            [
                "장르",
                "출시 연도",
                "단일 게임 수",
                "멀티 게임 수",
                "단일 실제 평가 수 중앙값",
                "멀티 실제 평가 수 중앙값",
                "로그 중앙값 차이",
            ]
        ]
        st.dataframe(
            display_table.style.format(
                {
                    "단일 실제 평가 수 중앙값": "{:,.1f}",
                    "멀티 실제 평가 수 중앙값": "{:,.1f}",
                    "로그 중앙값 차이": "{:+.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.info("음수인 조건은 초기 단일의 중앙값이 더 높은 예외 사례입니다. 개별 게임의 브랜드·마케팅 등 외부 요인을 함께 확인해야 합니다.")


def show_representative_tab(dataframe):
    st.subheader("같은 장르와 출시 연도에서 평가 참여도가 높은·보통·낮은 대표 게임은 무엇인가?")
    st.caption("현재 조건 안에서 장르와 출시 연도를 하나씩 선택하여 두 플랫폼 그룹을 비교합니다.")
    genre_dataframe = make_genre_dataframe(dataframe)

    if genre_dataframe.empty:
        st.warning("현재 조건에서 선택할 수 있는 장르가 없습니다.")
        return

    selector_column1, selector_column2 = st.columns(2)
    selected_genre = selector_column1.selectbox(
        "대표 사례 장르",
        sorted(genre_dataframe["genre"].unique()),
        key="representative_genre",
    )
    genre_data = genre_dataframe[
        genre_dataframe["genre"] == selected_genre
    ].copy()
    selected_year = selector_column2.selectbox(
        "대표 사례 출시 연도",
        sorted(genre_data["release_year"].unique(), reverse=True),
        key="representative_year",
    )
    selected_data = genre_data[
        genre_data["release_year"] == selected_year
    ].drop_duplicates(subset="id")

    group_counts = selected_data.groupby("initial_platform_group")["id"].nunique()
    metric_columns = st.columns(3)
    metric_columns[0].metric("선택 게임", f"{selected_data['id'].nunique():,}개")
    metric_columns[1].metric("초기 단일", f"{group_counts.get('single', 0):,}개")
    metric_columns[2].metric("초기 멀티", f"{group_counts.get('multi', 0):,}개")

    st.markdown("#### 이용자 평점과 평가 참여도")
    figure, axis = plt.subplots(figsize=(10, 5.5))
    sns.scatterplot(
        data=selected_data,
        x="rating",
        y="rating_count_log",
        hue="initial_platform_group",
        style="initial_platform_group",
        hue_order=PLATFORM_ORDER,
        style_order=PLATFORM_ORDER,
        palette=PLATFORM_COLORS,
        s=85,
        alpha=0.75,
        ax=axis,
    )
    axis.set_xlabel("IGDB 이용자 평점")
    axis.set_ylabel("평가 수 로그값")
    axis.legend(title="초기 플랫폼 그룹", labels=["초기 단일", "초기 멀티"])

    label_data = (
        selected_data.sort_values("rating_count_log", ascending=False)
        .groupby("initial_platform_group")
        .head(3)
    )
    for _, row in label_data.iterrows():
        axis.annotate(
            row["name"],
            (row["rating"], row["rating_count_log"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    st.pyplot(figure, width="stretch")
    plt.close(figure)

    st.markdown("#### 플랫폼 그룹별 대표 게임")
    representatives = make_representative_games(selected_data)
    if representatives.empty:
        st.warning("대표 게임을 선택할 데이터가 없습니다.")
        return

    representatives = add_group_labels(representatives)
    display_table = representatives.rename(
        columns={
            "name": "게임명",
            "case": "대표 구간",
            "rating": "이용자 평점",
            "rating_count": "평가 수",
            "initial_platform_count_30d": "초기 플랫폼 수",
        }
    )[
        [
            "플랫폼 그룹",
            "대표 구간",
            "게임명",
            "이용자 평점",
            "평가 수",
            "초기 플랫폼 수",
        ]
    ]
    st.dataframe(
        display_table.style.format(
            {"이용자 평점": "{:.1f}", "평가 수": "{:,.0f}"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.info("대표 사례는 현재 선택 조건 내부의 상대적 위치를 보여주며, 게임별 성과의 원인을 설명하는 목록은 아닙니다.")


def initialize_filter_state(min_year, max_year):
    if "year_range" not in st.session_state:
        st.session_state["year_range"] = (min_year, max_year)
    if "selected_genres" not in st.session_state:
        st.session_state["selected_genres"] = []
    if "min_sample" not in st.session_state:
        st.session_state["min_sample"] = 10


def show_sidebar(dataframe):
    min_year = int(dataframe["release_year"].min())
    max_year = int(dataframe["release_year"].max())
    all_genres = sorted(make_genre_dataframe(dataframe)["genre"].unique())
    initialize_filter_state(min_year, max_year)

    st.sidebar.header("공통 분석 조건")
    if st.sidebar.button("필터 초기화", width="stretch"):
        st.session_state["year_range"] = (min_year, max_year)
        st.session_state["selected_genres"] = []
        st.session_state["min_sample"] = 10
        st.rerun()

    year_range = st.sidebar.slider(
        "출시 연도",
        min_value=min_year,
        max_value=max_year,
        key="year_range",
    )
    selected_genres = st.sidebar.multiselect(
        "장르 (하나 이상 포함)",
        all_genres,
        key="selected_genres",
    )
    min_sample = st.sidebar.slider(
        "그룹별 최소 표본 수",
        min_value=5,
        max_value=30,
        key="min_sample",
        help="장르×출시 연도 비교에서 초기 단일과 초기 멀티 각각에 적용됩니다.",
    )

    filtered = filter_data(dataframe, year_range, selected_genres)
    st.sidebar.metric("현재 고유 게임 수", f"{filtered['id'].nunique():,}개")
    st.sidebar.caption("플랫폼 그룹은 비교 왜곡을 막기 위해 공통 필터에서 제외했습니다.")

    return filtered, min_sample


def main():
    st.set_page_config(
        page_title="IGDB 플랫폼 출시 범위 분석",
        page_icon="🎮",
        layout="wide",
    )
    sns.set_theme(style="whitegrid", font_scale=0.95)
    koreanize_matplotlib.koreanize()

    st.title("초기 플랫폼 출시 범위에 따른 IGDB 평가 참여 패턴 분석")
    st.write(
        "2017~2025년 출시 게임 중 IGDB 이용자 평점과 평가 수가 존재하는 게임을 대상으로, "
        "출시 후 30일 이내 플랫폼 범위와 누적 평가 참여도의 연관성을 살펴봅니다."
    )
    st.warning(
        "평가 수는 판매량이 아닌 공개 관심도의 대리지표입니다. 이 대시보드는 인과 효과가 아니라 "
        "관찰된 차이와 반복성, 예외 조건을 보여줍니다.",
        icon="⚠️",
    )

    try:
        dataframe = load_data(DATA_FILE)
    except FileNotFoundError:
        st.error(f"전처리 데이터 파일을 찾을 수 없습니다: {DATA_FILE}")
        st.stop()
    except ValueError as error:
        st.error(str(error))
        st.stop()

    filtered, min_sample = show_sidebar(dataframe)
    if filtered.empty:
        st.warning("현재 필터 조건에 해당하는 게임이 없습니다. 사이드바 조건을 변경해 주세요.")
        st.stop()

    tabs = st.tabs(
        ["1. 전체 비교", "2. 반복성", "3. 플랫폼 수", "4. 예외 조건", "5. 대표 게임"],
        key="analysis_tabs",
        on_change="rerun",
    )

    if tabs[0].open:
        with tabs[0]:
            show_overall_tab(filtered)
    if tabs[1].open:
        with tabs[1]:
            show_consistency_tab(filtered, min_sample)
    if tabs[2].open:
        with tabs[2]:
            show_platform_count_tab(filtered)
    if tabs[3].open:
        with tabs[3]:
            show_exception_tab(filtered, min_sample)
    if tabs[4].open:
        with tabs[4]:
            show_representative_tab(filtered)

    st.divider()
    st.caption(
        "데이터: IGDB API 수집·전처리 결과 · 초기 플랫폼: 최초 출시 후 30일 이내 출시 플랫폼 · "
        "평가 참여도: log1p(rating_count)"
    )


if __name__ == "__main__":
    main()
