import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================================================
# 1. 기본 설정
# =========================================================
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

# 50년 학습
학습50_시작 = 1956
학습50_끝 = 2005

# 100년 학습
학습100_시작 = 1906
학습100_끝 = 2005

# 공통 테스트 데이터
테스트_시작 = 2006
테스트_끝 = 2025


# =========================================================
# 2. 데이터 불러오기
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    df.columns = df.columns.str.strip()

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

    # 2025년까지
    df = df[
        df["날짜"].dt.year <= 기준연도
    ].copy()

    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온과 관측일수
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    # 1년에 300일 이상 관측된 연도만 사용
    yearly = yearly[
        yearly["관측일수"] >= 최소관측일수
    ].copy()

    yearly = (
        yearly
        .sort_values("연도")
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


# =========================================================
# 3. 회귀 함수
# =========================================================
def 회귀모델_만들기(data):

    x = data["연도"].to_numpy(dtype=float)
    y = data["연평균기온"].to_numpy(dtype=float)

    x변환 = x - 회귀기준연도

    slope, intercept = np.polyfit(
        x변환,
        y,
        1
    )

    return slope, intercept


def 예측하기(years, slope, intercept):

    years = np.asarray(
        years,
        dtype=float
    )

    x = years - 회귀기준연도

    return slope * x + intercept


def 평가하기(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    mae = np.mean(
        np.abs(actual - predicted)
    )

    mse = np.mean(
        (actual - predicted) ** 2
    )

    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - ss_res / ss_tot

    return mae, mse, r2


def 상관계수구하기(data):

    if len(data) < 2:
        return np.nan

    return np.corrcoef(
        data["연도"],
        data["연평균기온"]
    )[0, 1]


# =========================================================
# 4. 전체 데이터
# =========================================================
전체연도 = yearly["연도"].to_numpy(
    dtype=float
)

전체기온 = yearly["연평균기온"].to_numpy(
    dtype=float
)

전체기울기, 전체절편 = 회귀모델_만들기(
    yearly
)

전체예측 = 예측하기(
    전체연도,
    전체기울기,
    전체절편
)

전체_MAE, 전체_MSE, 전체_R2 = 평가하기(
    전체기온,
    전체예측
)

전체상관계수 = 상관계수구하기(
    yearly
)

전체_100년변화 = 전체기울기 * 100

전체시작연도 = int(
    yearly["연도"].min()
)

전체끝연도 = int(
    yearly["연도"].max()
)

전체연도수 = len(yearly)


# =========================================================
# 5. 학습 데이터 만들기
# =========================================================
학습50 = yearly[
    (yearly["연도"] >= 학습50_시작)
    & (yearly["연도"] <= 학습50_끝)
].copy()

학습100 = yearly[
    (yearly["연도"] >= 학습100_시작)
    & (yearly["연도"] <= 학습100_끝)
].copy()


# =========================================================
# 6. 테스트 데이터 만들기
# =========================================================
테스트 = yearly[
    (yearly["연도"] >= 테스트_시작)
    & (yearly["연도"] <= 테스트_끝)
].copy()

테스트 = (
    테스트
    .sort_values("연도")
    .reset_index(drop=True)
)


# =========================================================
# 7. 50년 / 100년 모델
# =========================================================
기울기50, 절편50 = 회귀모델_만들기(
    학습50
)

기울기100, 절편100 = 회귀모델_만들기(
    학습100
)

훈련상관50 = 상관계수구하기(
    학습50
)

훈련상관100 = 상관계수구하기(
    학습100
)

기온변화50 = 기울기50 * 100
기온변화100 = 기울기100 * 100


# =========================================================
# 8. 테스트 예측
# =========================================================
테스트연도 = 테스트["연도"].to_numpy(
    dtype=float
)

테스트실제기온 = 테스트["연평균기온"].to_numpy(
    dtype=float
)

테스트예측50 = 예측하기(
    테스트연도,
    기울기50,
    절편50
)

테스트예측100 = 예측하기(
    테스트연도,
    기울기100,
    절편100
)

mae50, mse50, r2_50 = 평가하기(
    테스트실제기온,
    테스트예측50
)

mae100, mse100, r2_100 = 평가하기(
    테스트실제기온,
    테스트예측100
)


# =========================================================
# 9. 최근 20년 회귀
# =========================================================
최근20년_시작 = 전체끝연도 - 19

최근20년 = yearly[
    yearly["연도"] >= 최근20년_시작
].copy()

최근20년 = (
    최근20년
    .sort_values("연도")
    .reset_index(drop=True)
)

최근20년_연도 = 최근20년["연도"].to_numpy(
    dtype=float
)

최근20년_기온 = 최근20년["연평균기온"].to_numpy(
    dtype=float
)

최근20년_기울기, 최근20년_절편 = 회귀모델_만들기(
    최근20년
)

최근20년_100년변화 = 최근20년_기울기 * 100

최근20년_상관계수 = 상관계수구하기(
    최근20년
)


# =========================================================
# 10. 제목
# =========================================================
st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연평균 기온 데이터를 이용해 "
    "선형회귀 모델을 만들고, 학습 기간에 따른 "
    "기울기와 예측 성능을 비교합니다."
)


# =========================================================
# 11. ① 데이터 개요
# =========================================================
st.divider()

st.header("① 데이터 개요")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "분석 시작 연도",
        f"{전체시작연도}년"
    )

with col2:
    st.metric(
        "분석 마지막 연도",
        f"{전체끝연도}년"
    )

with col3:
    st.metric(
        "사용 연도 수",
        f"{전체연도수}년"
    )

st.caption(
    "연간 관측일수가 300일 이상인 연도만 분석에 사용했습니다."
)


# =========================================================
# 12. ② 전체 데이터 회귀 분석
# =========================================================
st.divider()

st.header("② 전체 데이터에 대한 선형회귀")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "100년당 기온 변화",
        f"{전체_100년변화:+.2f} ℃"
    )

with col2:

    st.metric(
        "상관계수",
        f"{전체상관계수:.3f}"
    )

with col3:

    st.metric(
        "R²",
        f"{전체_R2:.3f}"
    )


# 전체 회귀 그래프
fig_all = go.Figure()

fig_all.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=6
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

회귀선연도 = np.arange(
    전체시작연도,
    2101
)

회귀선기온 = 예측하기(
    회귀선연도,
    전체기울기,
    전체절편
)

fig_all.add_trace(
    go.Scatter(
        x=회귀선연도,
        y=회귀선기온,
        mode="lines",
        name="전체 데이터 회귀선",
        line=dict(
            width=4
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀선: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig_all.update_layout(
    height=550,
    xaxis_title="연도",
    yaxis_title="연평균기온(℃)",
    template="plotly_white",
    hovermode="closest"
)

st.plotly_chart(
    fig_all,
    use_container_width=True
)


# =========================================================
# 13. ③ 50년 학습과 100년 학습 비교
# =========================================================
st.divider()

st.header("③ 50년 학습과 100년 학습 비교")

col1, col2 = st.columns(2)

# ---------------------------------------------------------
# 최근 50년
# ---------------------------------------------------------
with col1:

    st.subheader("최근 50년 학습")

    st.metric(
        "100년당 기온 변화",
        f"{기온변화50:.2f} ℃"
    )

    st.metric(
        "훈련 데이터 상관계수",
        f"{훈련상관50:.3f}"
    )

    st.caption(
        f"학습 기간: {학습50_시작}~{학습50_끝}"
    )

    st.caption(
        f"학습 연도 수: {len(학습50)}년"
    )


# ---------------------------------------------------------
# 최근 100년
# ---------------------------------------------------------
with col2:

    st.subheader("최근 100년 학습")

    st.metric(
        "100년당 기온 변화",
        f"{기온변화100:.2f} ℃"
    )

    st.metric(
        "훈련 데이터 상관계수",
        f"{훈련상관100:.3f}"
    )

    st.caption(
        f"학습 기간: {학습100_시작}~{학습100_끝}"
    )

    st.caption(
        f"학습 연도 수: {len(학습100)}년"
    )


# ---------------------------------------------------------
# 학습 기간에 따른 회귀선 비교
# ---------------------------------------------------------
st.subheader("📈 학습 기간에 따른 회귀선 비교")

fig_compare = go.Figure()

# 실제 데이터
fig_compare.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=5,
            opacity=0.5
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 50년 회귀선
비교연도50 = np.linspace(
    학습50_시작,
    테스트_끝,
    200
)

비교기온50 = 예측하기(
    비교연도50,
    기울기50,
    절편50
)

fig_compare.add_trace(
    go.Scatter(
        x=비교연도50,
        y=비교기온50,
        mode="lines",
        name="최근 50년 학습 회귀선",
        line=dict(
            width=4
        ),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "50년 회귀선: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 100년 회귀선
비교연도100 = np.linspace(
    학습100_시작,
    테스트_끝,
    300
)

비교기온100 = 예측하기(
    비교연도100,
    기울기100,
    절편100
)

fig_compare.add_trace(
    go.Scatter(
        x=비교연도100,
        y=비교기온100,
        mode="lines",
        name="최근 100년 학습 회귀선",
        line=dict(
            width=4,
            dash="dash"
        ),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "100년 회귀선: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 테스트 시작점 표시
fig_compare.add_vline(
    x=테스트_시작,
    line_width=2,
    line_dash="dot",
    annotation_text="테스트 시작"
)

fig_compare.update_layout(
    height=550,
    template="plotly_white",
    xaxis=dict(
        title="연도",
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
    )
)

st.plotly_chart(
    fig_compare,
    use_container_width=True
)


# =========================================================
# 14. ④ 테스트 데이터 예측 성능
# =========================================================
st.divider()

st.header("④ 테스트 데이터 예측 성능")

st.write(
    "50년 학습 모델과 100년 학습 모델이 "
    "공통 테스트 데이터인 2006~2025년의 "
    "연평균기온을 얼마나 잘 예측하는지 평가합니다."
)


# ---------------------------------------------------------
# 50년 모델 / 100년 모델 카드
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:

    st.subheader("🔵 50년 학습 모델")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "MAE",
            f"{mae50:.3f} ℃"
        )

    with c2:
        st.metric(
            "MSE",
            f"{mse50:.3f}"
        )

    with c3:
        st.metric(
            "R²",
            f"{r2_50:.3f}"
        )


with col2:

    st.subheader("🟠 100년 학습 모델")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "MAE",
            f"{mae100:.3f} ℃"
        )

    with c2:
        st.metric(
            "MSE",
            f"{mse100:.3f}"
        )

    with c3:
        st.metric(
            "R²",
            f"{r2_100:.3f}"
        )


# ---------------------------------------------------------
# 비교표
# ---------------------------------------------------------
st.subheader("📋 모델별 비교")

비교표 = pd.DataFrame(
    {
        "모델": [
            "전체 데이터",
            "최근 50년 학습",
            "최근 100년 학습"
        ],

        "학습 기간": [
            f"{전체시작연도}~{전체끝연도}",
            f"{학습50_시작}~{학습50_끝}",
            f"{학습100_시작}~{학습100_끝}"
        ],

        "평가 데이터": [
            "학습 데이터와 동일",
            f"{테스트_시작}~{테스트_끝}",
            f"{테스트_시작}~{테스트_끝}"
        ],

        "학습 연도 수": [
            전체연도수,
            len(학습50),
            len(학습100)
        ],

        "상관계수": [
            전체상관계수,
            훈련상관50,
            훈련상관100
        ],

        "100년당 기온 변화(℃)": [
            전체_100년변화,
            기온변화50,
            기온변화100
        ],

        "MAE": [
            전체_MAE,
            mae50,
            mae100
        ],

        "MSE": [
            전체_MSE,
            mse50,
            mse100
        ],

        "R²": [
            전체_R2,
            r2_50,
            r2_100
        ]
    }
)

비교표["상관계수"] = 비교표[
    "상관계수"
].round(3)

비교표["100년당 기온 변화(℃)"] = 비교표[
    "100년당 기온 변화(℃)"
].round(2)

비교표["MAE"] = 비교표[
    "MAE"
].round(3)

비교표["MSE"] = 비교표[
    "MSE"
].round(3)

비교표["R²"] = 비교표[
    "R²"
].round(3)

st.dataframe(
    비교표,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# 어느 모델이 좋은지
# ---------------------------------------------------------
st.info(
    f"""
    **해석 기준**

    • MAE: 실제값과 예측값의 평균적인 차이 → **작을수록 좋음**

    • MSE: 오차를 제곱하여 평균한 값 → **작을수록 좋음**

    • R²: 모델이 실제 기온 변화를 얼마나 설명하는지 나타내는 값 → **클수록 좋음**
    """
)


# =========================================================
# 15. ⑤ 최근 20년 실제 기온과 예측 기온
# =========================================================
st.divider()

st.header("⑤ 최근 20년의 실제 기온과 예측 기온")

st.write(
    "공통 테스트 데이터인 2006~2025년의 실제 연평균기온과 "
    "50년 학습 모델, 100년 학습 모델의 예측값을 비교합니다."
)


# ---------------------------------------------------------
# 최근 20년 표
# ---------------------------------------------------------
최근20_표 = 테스트[
    [
        "연도",
        "연평균기온"
    ]
].copy()

최근20_표[
    "50년 학습 모델 예측"
] = 테스트예측50

최근20_표[
    "100년 학습 모델 예측"
] = 테스트예측100

최근20_표[
    "50년 모델 오차"
] = (
    최근20_표["연평균기온"]
    - 최근20_표["50년 학습 모델 예측"]
)

최근20_표[
    "100년 모델 오차"
] = (
    최근20_표["연평균기온"]
    - 최근20_표["100년 학습 모델 예측"]
)

최근20_표["연평균기온"] = (
    최근20_표["연평균기온"].round(2)
)

최근20_표["50년 학습 모델 예측"] = (
    최근20_표["50년 학습 모델 예측"].round(2)
)

최근20_표["100년 학습 모델 예측"] = (
    최근20_표["100년 학습 모델 예측"].round(2)
)

최근20_표["50년 모델 오차"] = (
    최근20_표["50년 모델 오차"].round(2)
)

최근20_표["100년 모델 오차"] = (
    최근20_표["100년 모델 오차"].round(2)
)

최근20_표.columns = [
    "연도",
    "실제 평균기온(℃)",
    "50년 학습 모델 예측(℃)",
    "100년 학습 모델 예측(℃)",
    "50년 모델 오차(℃)",
    "100년 모델 오차(℃)"
]

st.dataframe(
    최근20_표,
    use_container_width=True,
    hide_index=True,
    height=760
)


# =========================================================
# 16. 최근 20년 실제값과 예측값 그래프
# =========================================================
st.subheader("📈 최근 20년 실제 기온과 예측 기온")

fig_test = go.Figure()

# 실제값
fig_test.add_trace(
    go.Scatter(
        x=테스트연도,
        y=테스트실제기온,
        mode="lines+markers",
        name="실제 평균기온",
        line=dict(
            width=3
        ),
        marker=dict(
            size=7
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "실제 평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 50년 예측
fig_test.add_trace(
    go.Scatter(
        x=테스트연도,
        y=테스트예측50,
        mode="lines+markers",
        name="50년 학습 모델",
        line=dict(
            width=3,
            dash="dash"
        ),
        marker=dict(
            size=5
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "50년 모델 예측: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 100년 예측
fig_test.add_trace(
    go.Scatter(
        x=테스트연도,
        y=테스트예측100,
        mode="lines+markers",
        name="100년 학습 모델",
        line=dict(
            width=3,
            dash="dot"
        ),
        marker=dict(
            size=5
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "100년 모델 예측: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig_test.update_layout(
    height=550,
    template="plotly_white",
    title={
        "text": "2006~2025년 실제 평균기온과 모델 예측",
        "x": 0.5
    },
    xaxis=dict(
        title="연도",
        dtick=2,
        showgrid=True
    ),
    yaxis=dict(
        title="평균기온(℃)",
        showgrid=True
    ),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)

st.plotly_chart(
    fig_test,
    use_container_width=True
)


# =========================================================
# 17. 50년 vs 100년 기울기와 성능 요약
# =========================================================
st.divider()

st.header("⑥ 50년 학습과 100년 학습의 차이")

col1, col2 = st.columns(2)

with col1:

    st.subheader("최근 50년 학습")

    st.metric(
        "기울기",
        f"{기울기50:.5f} ℃/년"
    )

    st.metric(
        "100년당 변화",
        f"{기온변화50:.2f} ℃"
    )

    st.metric(
        "테스트 MAE",
        f"{mae50:.3f} ℃"
    )

    st.metric(
        "테스트 R²",
        f"{r2_50:.3f}"
    )


with col2:

    st.subheader("최근 100년 학습")

    st.metric(
        "기울기",
        f"{기울기100:.5f} ℃/년"
    )

    st.metric(
        "100년당 변화",
        f"{기온변화100:.2f} ℃"
    )

    st.metric(
        "테스트 MAE",
        f"{mae100:.3f} ℃"
    )

    st.metric(
        "테스트 R²",
        f"{r2_100:.3f}"
    )


# =========================================================
# 18. 미래 기온 예측
# =========================================================
st.divider()

st.header("⑦ 미래 기온 예측")

예측연도 = st.slider(
    "예측할 연도",
    min_value=전체시작연도,
    max_value=2100,
    value=2026,
    step=1
)

예측기온 = 예측하기(
    예측연도,
    전체기울기,
    전체절편
)

st.metric(
    f"{예측연도}년 예상 연평균기온",
    f"{예측기온:.2f} ℃"
)

if 예측연도 > 전체끝연도:

    st.warning(
        "관측되지 않은 미래 연도이므로 "
        "회귀선을 연장한 추정값입니다."
    )


# =========================================================
# 19. 연도별 전체 데이터
# =========================================================
st.divider()

st.header("⑧ 연도별 평균기온 데이터")

표시데이터 = yearly[
    [
        "연도",
        "연평균기온",
        "관측일수"
    ]
].copy()

표시데이터["연평균기온"] = (
    표시데이터["연평균기온"].round(2)
)

표시데이터.columns = [
    "연도",
    "연평균기온(℃)",
    "관측일수"
]

st.dataframe(
    표시데이터,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "데이터 출처: 서울 기온 관측 데이터"
)
