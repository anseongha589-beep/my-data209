import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io


# =========================================================
# 1. 페이지 설정
# =========================================================
st.set_page_config(
    page_title="서울 100년 기온 데이터 정밀 분석",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. 화면 디자인
# =========================================================
st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f9fc;
    }

    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 5px;
    }

    .main-description {
        font-size: 16px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 15px;
    }

    .card-title {
        font-size: 15px;
        color: #6b7280;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .card-value {
        font-size: 27px;
        font-weight: 800;
        color: #111827;
    }

    .section-line {
        height: 1px;
        background-color: #e5e7eb;
        margin: 28px 0;
    }

    .section-title {
        font-size: 23px;
        font-weight: 750;
        color: #1f2937;
        margin-bottom: 4px;
    }

    .section-description {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 15px;
    }

    .tip-box {
        background-color: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 13px 16px;
        border-radius: 8px;
        margin-top: 10px;
        color: #7c2d12;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. 제목
# =========================================================
st.markdown(
    '<div class="main-title">🌡️ 서울 100년 연평균 기온 변화 분석</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-description">
    지난 100년 동안 서울의 연평균 기온 변화를 확인하고
    원본 데이터의 통계적 특성과 분포를 함께 분석하는 대시보드입니다.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 4. 데이터 불러오기
# =========================================================
@st.cache_data
def load_data():

    # 실제 CSV 주소를 사용하는 경우 이곳에 입력
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

        df = pd.read_csv(
            io.StringIO(html)
        )

        # ---------------------------------------------
        # 날짜 데이터가 있는 경우
        # ---------------------------------------------
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

        # ---------------------------------------------
        # 이미 연도별 데이터인 경우
        # ---------------------------------------------
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

        # =================================================
        # 백업 데이터
        # =================================================
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

        # 1908~2020년 전체 생성
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

        # 없는 연도는 보간
        df["평균기온"] = (
            df["평균기온"]
            .interpolate()
        )

        return df


# =========================================================
# 5. 데이터 처리
# =========================================================
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


    # =====================================================
    # 6. 전체 데이터 통계 계산
    # =====================================================
    orig_count = len(original_data)

    orig_mean = (
        original_data["평균기온"].mean()
    )

    orig_std = (
        original_data["평균기온"].std()
    )

    orig_min = (
        original_data["평균기온"].min()
    )

    orig_q1 = (
        original_data["평균기온"]
        .quantile(0.25)
    )

    orig_median = (
        original_data["평균기온"]
        .median()
    )

    orig_q3 = (
        original_data["평균기온"]
        .quantile(0.75)
    )

    orig_max = (
        original_data["평균기온"].max()
    )

    orig_iqr = orig_q3 - orig_q1


    # =====================================================
    # 7. 전체 기간 추세선
    # =====================================================
    if len(original_data) > 1:

        z_full = np.polyfit(
            original_data["연도"],
            original_data["평균기온"],
            1
        )

        slope_10y = z_full[0] * 10

    else:

        slope_10y = 0


    # =====================================================
    # 8. 전체 데이터 정밀 분석
    # =====================================================
    st.markdown(
        '<div class="section-title">📋 전체 데이터 정밀 분석</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        '서울 연평균 기온 데이터의 주요 통계값입니다.'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # 핵심 통계 카드
    # -----------------------------------------------------
    stat1, stat2, stat3, stat4 = st.columns(4)


    with stat1:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-title">📊 총 관측 데이터</div>
                <div class="card-value">{orig_count:,}개년</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with stat2:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-title">🌡️ 전체 평균기온</div>
                <div class="card-value">{orig_mean:.2f} °C</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with stat3:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-title">📈 10년당 상승 추세</div>
                <div class="card-value">{slope_10y:+.2f} °C</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with stat4:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-title">↔️ 기온 변동성</div>
                <div class="card-value">{orig_std:.2f} °C</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # 상세 통계표
    # -----------------------------------------------------
    detailed_summary_df = pd.DataFrame(
        {
            "데이터 요약 및 분포 지표": [
                "총 관측 데이터 개수 (Count)",
                "100년 전체 평균 기온 (Mean)",
                "기온 변동성 표준편차 (Std)",
                "역사상 최소 기온 (Min)",
                "하위 25% 기온 지점 (Q1)",
                "데이터 중간 기온값 (Median)",
                "상위 75% 기온 지점 (Q3)",
                "역사상 최대 기온 (Max)",
                "기온 밀집 구간 범위 (IQR)",
                "10년 단위 평균 기온 상승 추세"
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


    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # 9. 사이드바
    # =====================================================
    st.sidebar.header("📊 조회 설정")

    st.sidebar.write(
        "확인하고 싶은 연도 범위를 선택하세요."
    )


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


    st.sidebar.markdown("---")

    st.sidebar.info(
        "💡 그래프의 점에 마우스를 올리면 "
        "해당 연도의 평균기온을 확인할 수 있습니다."
    )


    # =====================================================
    # 10. 선택 기간 데이터
    # =====================================================
    filtered_data = original_data[
        (original_data["연도"] >= start_year)
        &
        (original_data["연도"] <= end_year)
    ].copy()


    # =====================================================
    # 11. 선택 기간 주요 지표
    # =====================================================
    if not filtered_data.empty:

        st.markdown(
            '<div class="section-title">📌 선택 기간 주요 지표</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="section-description">'
            f'{start_year}년부터 {end_year}년까지의 주요 기온 정보입니다.'
            f'</div>',
            unsafe_allow_html=True
        )


        col1, col2, col3 = st.columns(3)


        start_temp = (
            filtered_data["평균기온"]
            .iloc[0]
        )

        end_temp = (
            filtered_data["평균기온"]
            .iloc[-1]
        )

        diff = end_temp - start_temp


        with col1:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-title">
                        ⏰ {start_year}년 평균기온
                    </div>
                    <div class="card-value">
                        {start_temp:.1f} °C
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with col2:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-title">
                        ⏳ {end_year}년 평균기온
                    </div>
                    <div class="card-value">
                        {end_temp:.1f} °C
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with col3:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-title">
                        🌡️ 기간 동안의 기온 변화
                    </div>
                    <div class="card-value">
                        {diff:+.1f} °C
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # 12. 그래프 영역
    # =====================================================
    col_left, col_right = st.columns(
        [1.25, 0.75]
    )


    # =====================================================
    # 13. 왼쪽 - 연평균 기온 변화
    # =====================================================
    with col_left:

        st.markdown(
            '<div class="section-title">📈 연평균 기온 변화</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="section-description">'
            f'{start_year}년부터 {end_year}년까지의 기온 변화와 추세선'
            f'</div>',
            unsafe_allow_html=True
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
                    size=8
                ),

                customdata=filtered_data[
                    ["연도"]
                ],

                hovertemplate=(
                    "<b>📅 %{customdata[0]}년</b><br>"
                    "🌡️ 평균기온: %{y:.1f} °C"
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
                        color="#374151",
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

            height=550,

            plot_bgcolor="white",

            paper_bgcolor="white",

            hovermode="closest",

            xaxis=dict(
                title="연도",
                showgrid=True,
                gridcolor="#e5e7eb",
                dtick=10,
                zeroline=False
            ),

            yaxis=dict(
                title="평균기온 (°C)",
                showgrid=True,
                gridcolor="#e5e7eb",
                zeroline=False
            ),

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            ),

            margin=dict(
                l=50,
                r=20,
                t=70,
                b=50
            )
        )


        st.plotly_chart(
            fig_trend,
            use_container_width=True
        )


        st.markdown(
            """
            <div class="tip-box">
            💡 <b>마우스를 그래프의 빨간색 점에 올려보세요.</b><br>
            해당 연도의 정확한 평균기온이 표시됩니다.
            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # 14. 오른쪽 - 원본 데이터 분포
    # =====================================================
    with col_right:

        st.markdown(
            '<div class="section-title">📊 원본 데이터 분포</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            '전체 연평균 기온이 어느 구간에 많이 분포하는지 확인합니다.'
            '</div>',
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # 결측치 검사
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
                f"⚠️ {null_count}개의 결측치가 발견되었습니다."
            )


        # -------------------------------------------------
        # 핵심 분포 수치
        # -------------------------------------------------
        dist_col1, dist_col2 = st.columns(2)


        with dist_col1:

            st.metric(
                "🌡️ 평균",
                f"{orig_mean:.2f} °C"
            )


        with dist_col2:

            st.metric(
                "📍 중앙값",
                f"{orig_median:.2f} °C"
            )


        dist_col3, dist_col4 = st.columns(2)


        with dist_col3:

            st.metric(
                "⬇️ 최소",
                f"{orig_min:.1f} °C"
            )


        with dist_col4:

            st.metric(
                "⬆️ 최대",
                f"{orig_max:.1f} °C"
            )


        # -------------------------------------------------
        # 히스토그램
        # -------------------------------------------------
        st.markdown("#### 🌡️ 기온 분포")


        fig_hist = go.Figure()


        fig_hist.add_trace(
            go.Histogram(
                x=original_data["평균기온"],

                nbinsx=12,

                name="기온 분포",

                marker=dict(
                    color="#ff6b6b",
                    line=dict(
                        color="white",
                        width=1
                    )
                ),

                hovertemplate=(
                    "기온 구간: %{x:.1f} °C<br>"
                    "데이터 개수: %{y}개"
                    "<extra></extra>"
                )
            )
        )


        # 평균선
        fig_hist.add_vline(
            x=orig_mean,

            line_width=2,

            line_dash="dash",

            line_color="#374151",

            annotation_text=(
                f"평균 {orig_mean:.2f} °C"
            ),

            annotation_position="top"
        )


        fig_hist.update_layout(

            height=300,

            plot_bgcolor="white",

            paper_bgcolor="white",

            bargap=0.08,

            xaxis=dict(
                title="평균기온 (°C)",
                showgrid=True,
                gridcolor="#e5e7eb"
            ),

            yaxis=dict(
                title="데이터 개수",
                showgrid=True,
                gridcolor="#e5e7eb"
            ),

            showlegend=False,

            margin=dict(
                l=45,
                r=20,
                t=35,
                b=45
            )
        )


        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )


        # -------------------------------------------------
        # 박스플롯
        # -------------------------------------------------
        st.markdown("#### 📦 기온 범위 및 이상치")


        fig_box = go.Figure()


        fig_box.add_trace(
            go.Box(
                x=original_data["평균기온"],

                orientation="h",

                name="전체 데이터",

                boxpoints="outliers",

                marker=dict(
                    size=7,
                    color="#ff4b4b"
                ),

                line=dict(
                    width=2
                ),

                fillcolor="#ffe8e8",

                hovertemplate=(
                    "평균기온: %{x:.1f} °C"
                    "<extra></extra>"
                )
            )
        )


        # 평균선
        fig_box.add_vline(
            x=orig_mean,

            line_width=2,

            line_dash="dash",

            line_color="#374151",

            annotation_text=(
                f"평균 {orig_mean:.2f} °C"
            ),

            annotation_position="top"
        )


        fig_box.update_layout(

            height=180,

            plot_bgcolor="white",

            paper_bgcolor="white",

            xaxis=dict(
                title="평균기온 (°C)",
                showgrid=True,
                gridcolor="#e5e7eb"
            ),

            yaxis=dict(
                showticklabels=False
            ),

            showlegend=False,

            margin=dict(
                l=20,
                r=20,
                t=35,
                b=45
            )
        )


        st.plotly_chart(
            fig_box,
            use_container_width=True
        )


        # -------------------------------------------------
        # 분포 요약
        # -------------------------------------------------
        st.markdown(
            f"""
            <div class="tip-box">
            <b>📌 분포 요약</b><br><br>

            • 평균기온:
            <b>{orig_mean:.2f} °C</b><br>

            • 중앙값:
            <b>{orig_median:.2f} °C</b><br>

            • 전체 데이터의 50%가 포함된 범위:
            <b>{orig_q1:.2f} ~ {orig_q3:.2f} °C</b><br>

            • 전체 기온 범위:
            <b>{orig_min:.1f} ~ {orig_max:.1f} °C</b><br>

            • 사분위간 범위(IQR):
            <b>{orig_iqr:.2f} °C</b>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # 15. 전체 데이터 테이블
    # =====================================================
    st.markdown(
        '<div class="section-title">📄 원본 데이터</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        '선택한 기간의 실제 데이터를 확인할 수 있습니다.'
        '</div>',
        unsafe_allow_html=True
    )


    show_table = st.checkbox(
        "📊 전체 데이터 테이블 보기"
    )


    if show_table:

        st.dataframe(
            filtered_data.set_index("연도"),
            use_container_width=True,
            height=450
        )


# =========================================================
# 16. 오류 처리
# =========================================================
except Exception as e:

    st.error(
        f"데이터를 처리하는 중 오류가 발생했습니다: {e}"
    )
