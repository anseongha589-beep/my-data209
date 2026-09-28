
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==================================================
# 1. 기본 설정
# ==================================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

기준연도 = 2025
최소관측일수 = 300
회귀기준연도 = 1908


# ==================================================
# 2. 제목
# ==================================================
st.title("🌡️ 기온 예측기")

st.markdown(
    """
    서울의 과거 연평균기온을 분석하고,
    선형회귀를 이용해 원하는 연도의 예상 기온을 확인합니다.

    전체 기간의 기온 변화 추세와 최근 20년의 추세도 비교합니다.
    """
)


# ==================================================
# 3. 데이터 불러오기
# ==================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df.columns = df.columns.str.strip()

    # 날짜와 평균기온 정리
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    # 2025년까지의 자료만 사용
    df = df[df["날짜"].dt.year <= 기준연도].copy()

    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온과 관측일수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 연도 제외
    yearly = yearly[
        yearly["관측일수"] >= 최소관측일수
    ].copy()

    yearly = (
        yearly.sort_values("연도")
        .reset_index(drop=True)
    )

    return yearly


try:
    yearly = load_data()

except Exception as e:
    st.error("기온 데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


if len(yearly) < 2:
    st.error("회귀 분석에 사용할 데이터가 부족합니다.")
    st.stop()


# ==================================================
# 4. 전체 기간 회귀 분석
# ==================================================
전체연도 = yearly["연도"].to_numpy(dtype=float)
전체기온 = yearly["연평균기온"].to_numpy(dtype=float)

# 독립 변수: 1908년부터 지난 연수
전체x = 전체연도 - 회귀기준연도

전체기울기, 전체절편 = np.polyfit(
    전체x,
    전체기온,
    1
)

전체상관계수 = np.corrcoef(
    전체x,
    전체기온
)[0, 1]

# 1년당 기울기를 100년당 변화량으로 환산
전체_100년변화 = 전체기울기 * 100

전체시작연도 = int(yearly["연도"].min())
전체끝연도 = int(yearly["연도"].max())
전체연도수 = len(yearly)


# ==================================================
# 5. 최근 20년 회귀 분석
# ==================================================
# 마지막 관측 연도를 기준으로 20년 범위 설정
최근20년_시작기준 = 전체끝연도 - 19

최근20년 = yearly[
    yearly["연도"] >= 최근20년_시작기준
].copy()

최근20년 = 최근20년.sort_values("연도")

최근20년_연도수 = len(최근20년)

if 최근20년_연도수 >= 2:

    최근연도 = 최근20년["연도"].to_numpy(dtype=float)
    최근기온 = 최근20년["연평균기온"].to_numpy(dtype=float)

    최근x = 최근연도 - 회귀기준연도

    최근기울기, 최근절편 = np.polyfit(
        최근x,
        최근기온,
        1
    )

    최근상관계수 = np.corrcoef(
        최근x,
        최근기온
    )[0, 1]

    최근_100년변화 = 최근기울기 * 100

    최근시작연도 = int(최근20년["연도"].min())
    최근끝연도 = int(최근20년["연도"].max())

else:
    최근기울기 = np.nan
    최근절편 = np.nan
    최근상관계수 = np.nan
    최근_100년변화 = np.nan
    최근시작연도 = None
    최근끝연도 = None


# ==================================================
# 6. 기울기 표시 문구
# ==================================================
def 변화문구(value):
    if value > 0:
        return f"100년당 {value:+.2f}℃ 상승 추세"
    elif value < 0:
        return f"100년당 {value:+.2f}℃ 하강 추세"
    else:
        return "100년당 변화 없음"


# ==================================================
# 7. 전체 기간과 최근 20년 비교
# ==================================================
st.divider()
st.subheader("📈 100년당 기온 변화 비교")

st.markdown(
    """
    회귀 직선의 기울기에 100을 곱해
    100년 동안의 평균적인 기온 변화량으로 나타냅니다.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 전체 기간")

    st.metric(
        label="100년당 기온 변화",
        value=f"{전체_100년변화:+.2f} ℃"
    )

    st.markdown(
        f"**{전체시작연도}년 ~ {전체끝연도}년**"
    )

    st.caption(
        f"회귀에 사용한 연도: {전체연도수}개"
    )

    st.caption(
        변화문구(전체_100년변화)
    )

with col2:
    st.markdown("### 최근 20년")

    if 최근20년_연도수 >= 2:

        st.metric(
            label="100년당 기온 변화",
            value=f"{최근_100년변화:+.2f} ℃"
        )

        st.markdown(
            f"**{최근시작연도}년 ~ {최근끝연도}년**"
        )

        st.caption(
            f"회귀에 사용한 연도: {최근20년_연도수}개"
        )

        st.caption(
            변화문구(최근_100년변화)
        )

    else:
        st.warning(
            "최근 20년 자료가 부족해 비교할 수 없습니다."
        )


# ==================================================
# 8. 기울기 비교 설명
# ==================================================
if 최근20년_연도수 >= 2:

    기울기차이 = 최근_100년변화 - 전체_100년변화

    st.info(
        f"""
        **기울기 차이: {기울기차이:+.2f}℃ / 100년**

        최근 20년의 100년당 변화량에서
        전체 기간의 100년당 변화량을 뺀 값입니다.

        양수이면 최근 20년의 상승 추세가
        전체 기간의 상승 추세보다 큰 것이고,
        음수이면 더 작은 것입니다.
        """
    )


# ==================================================
# 9. 상관계수와 회귀식
# ==================================================
st.divider()
st.subheader("📊 전체 기간 회귀 분석")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "회귀에 사용한 연도",
        f"{전체연도수}개"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{전체시작연도}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{전체끝연도}년"
    )

st.markdown(
    f"**전체 기간 상관계수: {전체상관계수:.4f}**"
)

st.markdown(
    f"""
    독립 변수는 연도에서 {회귀기준연도}를 뺀 값입니다.

    - 전체 기간 회귀식:
    """
)

st.latex(
    rf"\hat{{y}} = {전체절편:.4f} "
    rf"+ ({전체기울기:.6f})x"
)

if 최근20년_연도수 >= 2:

    st.markdown("최근 20년 회귀식:")

    st.latex(
        rf"\hat{{y}} = {최근절편:.4f} "
        rf"+ ({최근기울기:.6f})x"
    )


# ==================================================
# 10. 예상 기온 계산
# ==================================================
def predict_temperature(year):
    x = year - 회귀기준연도
    return 전체기울기 * x + 전체절편


st.divider()
st.subheader("🔮 미래 기온 예측")

selected_year = st.slider(
    "예상 기온을 확인할 연도",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

predicted_temp = predict_temperature(selected_year)

st.markdown(
    f"### {selected_year}년 예상 연평균기온"
)

st.markdown(
    f"""
    <div style="
        background-color: #e8f3ff;
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 20px;
    ">
        <div style="
            font-size: 22px;
            color: #245580;
            margin-bottom: 8px;
        ">
            예상 평균기온
        </div>
        <div style="
            font-size: 48px;
            font-weight: bold;
            color: #1261a0;
        ">
            {predicted_temp:.2f} ℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if selected_year > 전체끝연도:
    st.warning(
        "실제 관측 자료가 없는 미래 연도입니다. "
        "전체 기간 회귀 직선을 연장한 추정값입니다."
    )

elif selected_year < 전체시작연도:
    st.warning(
        "회귀 분석에 사용한 기간보다 이전 연도입니다. "
        "회귀식을 과거로 연장한 추정값입니다."
    )


# ==================================================
# 11. 산점도와 회귀 직선
# ==================================================
st.divider()
st.subheader("📉 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 관측 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7,
            color="#2878B5",
            opacity=0.75
        ),
        customdata=yearly[["관측일수"]],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 전체 기간 회귀 직선
line_years = np.arange(1900, 2101)

전체회귀선 = (
    전체기울기 * (line_years - 회귀기준연도)
    + 전체절편
)

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=전체회귀선,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(
            color="#E4572E",
            width=3
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "전체 기간 회귀 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 최근 20년 회귀 직선
if 최근20년_연도수 >= 2:

    최근선_연도 = np.linspace(
        최근시작연도,
        최근끝연도,
        100
    )

    최근회귀선 = (
        최근기울기 * (최근선_연도 - 회귀기준연도)
        + 최근절편
    )

    fig.add_trace(
        go.Scatter(
            x=최근선_연도,
            y=최근회귀선,
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(
                color="#16A085",
                width=3,
                dash="dash"
            ),
            hovertemplate=(
                "연도: %{x:.0f}년<br>"
                "최근 20년 회귀 기온: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )

# 선택한 연도의 예상 기온
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name="선택한 연도 예상 기온",
        marker=dict(
            size=14,
            color="#F4B400",
            line=dict(
                color="black",
                width=1
            )
        ),
        hovertemplate=(
            "선택 연도: %{x}년<br>"
            "예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    height=600,
    template="plotly_white",
    title={
        "text": "서울 연평균기온과 회귀 직선 비교",
        "x": 0.5
    },
    xaxis=dict(
        title="연도",
        range=[1900, 2100],
        tickmode="linear",
        tick0=1900,
        dtick=10,
        showgrid=True
    ),
    yaxis=dict(
        title="연평균기온(℃)",
        showgrid=True
    ),
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),
    margin=dict(
        l=40,
        r=40,
        t=100,
        b=50
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==================================================
# 12. 회귀 분석 데이터 표
# ==================================================
st.divider()
st.subheader("🗂️ 회귀 분석에 사용한 연도별 데이터")

st.caption(
    "2025년까지의 자료 중 관측일이 300일 이상인 연도입니다."
)

display_df = yearly[
    ["연도", "연평균기온", "관측일수"]
].copy()

display_df["연평균기온"] = (
    display_df["연평균기온"].round(2)
)

display_df.columns = [
    "연도",
    "연평균기온(℃)",
    "관측일수"
]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "데이터 출처: 서울 기온 관측 데이터"
)
