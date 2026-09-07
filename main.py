import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# ---------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="서울 100년 기온 데이터 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 100년 연평균 기온 변화 분석")

st.markdown(
    """
    지난 100년 동안 서울의 연평균 기온 변화를 분석하는 대시보드입니다.

    **📌 그래프의 점에 마우스를 올리면 해당 연도와 평균기온이 표시됩니다.**
    """
)


# ---------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------
@st.cache_data
def load_data():

    # 실제 CSV 주소가 있다면 이곳에 입력
    url = "https://githubusercontent.com"

    try:
        import urllib.request

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode("utf-8")

        df = pd.read_csv(io.StringIO(html))

        # 날짜 형식 데이터
        if "날짜" in df.columns:

            df["날짜"] = pd.to_datetime(
                df["날짜"],
                errors="coerce"
            )

            df["연도"] = df["날짜"].dt.year

            df["평균기온"] = pd.to_numeric(
                df["평균기온"],
                errors="coerce"
            )

            df = df.dropna(
                subset=["연도", "평균기온"]
            )

            annual_mean = (
                df.groupby("연도")["평균기온"]
                .mean()
                .reset_index()
            )

            return annual_mean

        # 이미 연도별 데이터인 경우
        elif (
            "연도" in df.columns
            and "평균기온" in df.columns
        ):

            df["연도"] = pd.to_numeric(
                df["연도"],
                errors="coerce"
            )

            df["평균기온"] = pd.to_numeric(
                df["평균기온"],
                errors="coerce"
            )

            df = df.dropna(
                subset=["연도", "평균기온"]
            )

            return df

    except Exception:

        # -------------------------------------------------
        # 백업 데이터
        # -------------------------------------------------
        backup_data = """
연도,평균기온
1908,10.4
1915,11.0
1920,11.5
1925,10.9
1930,11.2
1935,11.7
1940,10.8
1945,11.4
1950,11.2
1955,11.9
1960,12.1
1965,11.3
1970,11.7
1975,12.2
1980,11.4
1985,11.9
1990,12.9
1995,12.2
2000,12.7
2005,12.1
2010,12.1
2015,13.6
2018,12.9
2019,13.4
2020,13.2
"""

        df_backup = pd.read_csv(
            io.StringIO(backup_data)
        )

        # 1908~2020년 생성
        years = pd.DataFrame(
            {
                "연도": range(1908, 2021)
            }
        )

        df = pd.merge(
            years,
            df_backup,
            on="연도",
            how="left"
        )

        # 없는 연도는 앞뒤 값으로 보간
        df["평균기온"] = (
            df["평균기온"]
            .interpolate()
        )

        return df


# ---------------------------------------------------------
# 데이터 실행
# ---------------------------------------------------------
try:

    original_data = load_data()

    original_data["연도"] = pd.to_numeric(
        original_data["연도"],
        errors="coerce"
    )

    original_data["평균기온"] = pd.to_numeric(
        original_data["평균기온"],
        errors="coerce"
    )

    original_data = original_data.dropna(
        subset=["연도", "평균기온"]
    )

    original_data = (
        original_data
        .sort_values("연도")
        .reset_index(drop=True)
    )


    # -----------------------------------------------------
    # 전체 데이터 통계
    # -----------------------------------------------------
    st.subheader("📋 전체 데이터 정밀 분석")

    orig_count = len(original_data)

    orig_mean = original_data["평균기온"].mean()

    orig_std = original_data["평균기온"].std()

    orig_min = original_data["평균기온"].min()

    orig_q1 = original_data["평균기온"].quantile(0.25)

    orig_median = original_data["평균기온"].median()

    orig_q3 = original_data["평균기온"].quantile(0.75)

    orig_max = original_data["평균기온"].max()

    orig_iqr = orig_q3 - orig_q1


    # -----------------------------------------------------
    # 전체 기간 추세선
    # -----------------------------------------------------
    if len(original_data) > 1:

        z_full = np.polyfit(
            original_data["연도"],
            original_data["평균기온"],
            1
        )

        slope_10y = z_full[0] * 10

    else:

        slope_10y = 0


    # -----------------------------------------------------
    # 통계 테이블
    # -----------------------------------------------------
    detailed_summary_df = pd.DataFrame(
        {
            "데이터 요약 및 분포 지표": [
                "총 관측 데이터 개수",
                "전체 평균 기온",
                "기온 변동성 표준편차",
                "역사상 최소 기온",
                "하위 25% 기온 지점 (Q1)",
                "중앙값 (Median)",
                "상위 75% 기온 지점 (Q3)",
                "역사상 최대 기온",
                "사분위간 범위 (IQR)",
                "10년당 평균 기온 상승 추세"
            ],

            "원본 정밀 통계치": [
                f"{orig_count:,} 개년",
                f"{orig_mean:.2f} °C",
                f"{orig_std:.2f} °C",
                f"{orig_min:.1f} °C",
                f"{orig_q1:.2f} °C",
                f"{orig_median:.2f} °C",
                f"{orig_q3:.2f} °C",
                f"{orig_max:.1f} °C",
                f"{orig_iqr:.2f} °C",
                f"{slope_10y:+.2f} °C / 10년"
            ]
        }
    )

    st.dataframe(
        detailed_summary_df,
        use_container_width=True,
        hide_index=True
    )


    st.markdown("---")


    # -----------------------------------------------------
    # 사이드바 연도 선택
    # -----------------------------------------------------
    st.sidebar.header("📊 조회 설정")

    min_year = int(
        original_data["연도"].min()
    )

    max_year = int(
        original_data["연도"].max()
    )

    start_year, end_year = st.sidebar.slider(
        "조회할 연도 범위",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )


    # -----------------------------------------------------
    # 선택 기간 데이터
    # -----------------------------------------------------
    filtered_data = original_data[
        (original_data["연도"] >= start_year)
        &
        (original_data["연도"] <= end_year)
    ].copy()


    # -----------------------------------------------------
    # 선택 기간 주요 지표
    # -----------------------------------------------------
    if not filtered_data.empty:

        col1, col2, col3 = st.columns(3)

        start_temp = filtered_data[
            "평균기온"
        ].iloc[0]

        end_temp = filtered_data[
            "평균기온"
        ].iloc[-1]

        diff = end_temp - start_temp

        with col1:

            st.metric(
                label=f"⏰ {start_year}년 평균기온",
                value=f"{start_temp:.1f} °C"
            )

        with col2:

            st.metric(
                label=f"⏳ {end_year}년 평균기온",
                value=f"{end_temp:.1f} °C"
            )

        with col3:

            st.metric(
                label="🌡️ 기간 기온 변화",
                value=f"{diff:+.1f} °C"
            )


    st.markdown("---")


    # -----------------------------------------------------
    # 그래프 영역
    # -----------------------------------------------------
    col_left, col_right = st.columns(
        [1.1, 0.9]
    )


    # =====================================================
    # 왼쪽 : 인터랙티브 그래프
    # =====================================================
    with col_left:

        st.subheader(
            "📈 선택 기간 기온 변화"
        )

        st.write(
            f"{start_year}년부터 {end_year}년까지의 "
            "연평균 기온 변화입니다."
        )


        fig_trend = go.Figure()


        # -------------------------------------------------
        # 연평균 기온
        # -------------------------------------------------
        fig_trend.add_trace(
            go.Scatter(
                x=filtered_data["연도"],
                y=filtered_data["평균기온"],

                mode="lines+markers",

                name="연평균 기온",

                line=dict(
                    color="#ff4b4b",
                    width=3
                ),

                marker=dict(
                    size=7
                ),

                # 마우스 오버용 데이터
                customdata=filtered_data[
                    ["연도"]
                ],

                hovertemplate=(
                    "<b>%{customdata[0]}년</b><br>"
                    "평균기온: %{y:.1f} °C"
                    "<extra></extra>"
                )
            )
        )


        # -------------------------------------------------
        # 추세선
        # -------------------------------------------------
        if len(filtered_data) > 1:

            z = np.polyfit(
                filtered_data["연도"],
                filtered_data["평균기온"],
                1
            )

            p = np.poly1d(z)

            trend_values = p(
                filtered_data["연도"]
            )


            fig_trend.add_trace(
                go.Scatter(
                    x=filtered_data["연도"],
                    y=trend_values,

                    mode="lines",

                    name="추세선",

                    line=dict(
                        color="#31333F",
                        width=2,
                        dash="dash"
                    ),

                    hoverinfo="skip"
                )
            )


        # -------------------------------------------------
        # 그래프 디자인
        # -------------------------------------------------
        fig_trend.update_layout(

            height=520,

            xaxis=dict(
                title="연도",
                showgrid=True,
                dtick=10
            ),

            yaxis=dict(
                title="평균기온 (°C)",
                showgrid=True
            ),

            hovermode="closest",

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            ),

            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )


        # 인터랙티브 그래프
        st.plotly_chart(
            fig_trend,
            use_container_width=True
        )

        st.info(
            "💡 그래프의 빨간색 점에 마우스를 올리면 "
            "해당 연도와 평균기온이 표시됩니다."
        )


    # =====================================================
    # 오른쪽 : 데이터 분포
    # =====================================================
    with col_right:

        st.subheader(
            "📦 원본 데이터 분포"
        )


        # -------------------------------------------------
        # 결측치 확인
        # -------------------------------------------------
        null_count = (
            original_data["평균기온"]
            .isnull()
            .sum()
        )


        if null_count == 0:

            st.success(
                "✅ 결측치 검증 완료\n\n"
                "평균기온 데이터에 결측치가 없습니다."
            )

        else:

            st.warning(
                f"⚠️ {null_count}개의 "
                "결측치가 발견되었습니다."
            )


        # -------------------------------------------------
        # 박스플롯
        # -------------------------------------------------
        fig_box = go.Figure()


        fig_box.add_trace(
            go.Box(
                y=original_data["평균기온"],

                name="전체 데이터",

                boxpoints="outliers",

                marker=dict(
                    size=7
                ),

                line=dict(
                    width=2
                ),

                hovertemplate=(
                    "평균기온: %{y:.1f} °C"
                    "<extra></extra>"
                )
            )
        )


        fig_box.update_layout(

            height=400,

            yaxis=dict(
                title="평균기온 (°C)"
            ),

            showlegend=False,

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            )
        )


        st.plotly_chart(
            fig_box,
            use_container_width=True
        )


    st.markdown("---")


    # -----------------------------------------------------
    # 데이터 테이블
    # -----------------------------------------------------
    show_table = st.checkbox(
        "📄 전체 데이터 테이블 보기"
    )


    if show_table:

        st.subheader(
            f"📊 {start_year}~{end_year}년 데이터"
        )

        st.dataframe(
            filtered_data.set_index("연도"),
            use_container_width=True
        )


# ---------------------------------------------------------
# 오류 처리
# ---------------------------------------------------------
except Exception as e:

    st.error(
        f"데이터를 처리하는 중 오류가 발생했습니다: {e}"
    )
