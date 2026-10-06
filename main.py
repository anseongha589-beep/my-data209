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

# 학습 / 테스트 기간
학습50_시작 = 1956
학습50_끝 = 2005

학습100_시작 = 1906
학습100_끝 = 2005

테스트_시작 = 2006
테스트_끝 = 2025


# ==================================================
# 2. 제목
# ==================================================
st.title("🌡️ 기온 예측기")

st.markdown(
    """
    서울의 과거 연평균기온을 분석하고 선형회귀 모델을 이용해
    기온 변화 추세와 미래 기온을 예측합니다.

    또한 과거 데이터를 **학습 데이터**와 **테스트 데이터**로 나누어
    회귀 모델이 최근 기온을 얼마나 잘 예측하는지 평가합니다.
    """
)


# ==================================================
# 3. 데이터 불러오기
# ==================================================
@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    df.columns = df.columns.str.strip()

    # 날짜 정리
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    # 2025년까지 사용
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

    # 관측일이 300일 미만인 연도 제외
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


# ==================================================
# 4. 기본 데이터 확인
# ==================================================
전체연도 = yearly["연도"].to_numpy(dtype=float)
전체기온 = yearly["연평균기온"].to_numpy(dtype=float)

전체시작연도 = int(yearly["연도"].min())
전체끝연도 = int(yearly["연도"].max())
전체연도수 = len(yearly)


# ==================================================
# 5. 전체 기간 회귀 분석
# ==================================================
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

전체_100년변화 = 전체기울기 * 100


# ==================================================
# 6. 최근 20년 회귀 분석
# ==================================================
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
# 7. 변화 문구
# ==================================================
def 변화문구(value):

    if value > 0:
        return f"100년당 {value:+.2f}℃ 상승 추세"

    elif value < 0:
        return f"100년당 {value:+.2f}℃ 하강 추세"

    else:
        return "100년당 변화 없음"


# ==================================================
# 8. 학습 데이터 / 테스트 데이터 분리
# ==================================================
학습50 = yearly[
    (yearly["연도"] >= 학습50_시작)
    & (yearly["연도"] <= 학습50_끝)
].copy()

학습100 = yearly[
    (yearly["연도"] >= 학습100_시작)
    & (yearly["연도"] <= 학습100_끝)
].copy()

테스트 = yearly[
    (yearly["연도"] >= 테스트_시작)
    & (yearly["연도"] <= 테스트_끝)
].copy()


# ==================================================
# 9. 회귀 모델 함수
# ==================================================
def 회귀모델_만들기(data):

    x = data["연도"].to_numpy(dtype=float)
    y = data["연평균기온"].to_numpy(dtype=float)

    x_변환 = x - 회귀기준연도

    기울기, 절편 = np.polyfit(
        x_변환,
        y,
        1
    )

    return 기울기, 절편


def 예측하기(years, slope, intercept):

    x = np.asarray(years, dtype=float) - 회귀기준연도

    return slope * x + intercept


# ==================================================
# 10. 평가 지표 함수
# ==================================================
def 평가하기(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    # MAE
    mae = np.mean(
        np.abs(actual - predicted)
    )

    # MSE
    mse = np.mean(
        (actual - predicted) ** 2
    )

    # R²
    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - (ss_res / ss_tot)

    return mae, mse, r2


# ==================================================
# 11. 50년 / 100년 회귀 모델 만들기
# ==================================================
if len(학습50) >= 2:
    기울기50, 절편50 = 회귀모델_만들기(학습50)
else:
    기울기50 = np.nan
    절편50 = np.nan


if len(학습100) >= 2:
    기울기100, 절편100 = 회귀모델_만들기(학습100)
else:
    기울기100 = np.nan
    절편100 = np.nan


# ==================================================
# 12. 테스트 데이터 예측 및 평가
# ==================================================
if (
    len(테스트) >= 2
    and not np.isnan(기울기50)
    and not np.isnan(기울기100)
):

    테스트연도 = 테스트["연도"].to_numpy(dtype=float)
    테스트실제기온 = 테스트["연평균기온"].to_numpy(dtype=float)

    # 50년 학습 모델 예측
    테스트예측50 = 예측하기(
        테스트연도,
        기울기50,
        절편50
    )

    # 100년 학습 모델 예측
    테스트예측100 = 예측하기(
        테스트연도,
        기울기100,
        절편100
    )

    # 평가
    mae50, mse50, r2_50 = 평가하기(
        테스트실제기온,
        테스트예측50
    )

    mae100, mse100, r2_100 = 평가하기(
        테스트실제기온,
        테스트예측100
    )

else:

    테스트연도 = np.array([])
    테스트실제기온 = np.array([])

    테스트예측50 = np.array([])
    테스트예측100 = np.array([])

    mae50 = np.nan
    mse50 = np.nan
    r2_50 = np.nan

    mae100 = np.nan
    mse100 = np.nan
    r2_100 = np.nan


# ==================================================
# 13. 기존 전체 기간 / 최근 20년 기울기 비교
# ==================================================
st.divider()

st.subheader("📈 100년당 기온 변화 비교")

st.markdown(
    """
    회귀선의 기울기를 100년 기준으로 환산하여
    전체 기간과 최근 20년의 기온 변화 추세를 비교합니다.
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


if 최근20년_연도수 >= 2:

    기울기차이 = 최근_100년변화 - 전체_100년변화

    st.info(
        f"""
        **기울기 차이: {기울기차이:+.2f}℃ / 100년**

        최근 20년의 100년당 변화량에서
        전체 기간의 100년당 변화량을 뺀 값입니다.

        양수이면 최근 20년의 상승 추세가
        전체 기간보다 크다는 뜻이고,
        음수이면 더 작다는 뜻입니다.
        """
    )


# ==================================================
# 14. 핵심 분석
# ==================================================
st.divider()

st.subheader("🎯 50년 학습 vs 100년 학습")

st.markdown(
    f"""
    두 모델은 모두 **2006~2025년을 공통 테스트 데이터**로 사용합니다.

    - **50년 모델:** {학습50_시작}~{학습50_끝}년 학습
    - **100년 모델:** {학습100_시작}~{학습100_끝}년 학습
    - **테스트:** {테스트_시작}~{테스트_끝}년
    """
)


# ==================================================
# 15. 학습 데이터 개수 확인
# ==================================================
col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "50년 학습 데이터",
        f"{len(학습50)}개 연도"
    )

with col2:

    st.metric(
        "100년 학습 데이터",
        f"{len(학습100)}개 연도"
    )

with col3:

    st.metric(
        "공통 테스트 데이터",
        f"{len(테스트)}개 연도"
    )


# ==================================================
# 16. 회귀선 기울기 비교
# ==================================================
st.subheader("📐 회귀선 기울기 비교")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 최근 50년 학습")

    if not np.isnan(기울기50):

        st.metric(
            "100년당 기온 변화",
            f"{기울기50 * 100:+.2f} ℃"
        )

        st.caption(
            f"1년당 기울기: {기울기50:.6f} ℃"
        )

        st.caption(
            f"학습 기간: {학습50_시작}~{학습50_끝}년"
        )

    else:

        st.warning(
            "50년 학습 데이터가 부족합니다."
        )


with col2:

    st.markdown("### 최근 100년 학습")

    if not np.isnan(기울기100):

        st.metric(
            "100년당 기온 변화",
            f"{기울기100 * 100:+.2f} ℃"
        )

        st.caption(
            f"1년당 기울기: {기울기100:.6f} ℃"
        )

        st.caption(
            f"학습 기간: {학습100_시작}~{학습100_끝}년"
        )

    else:

        st.warning(
            "100년 학습 데이터가 부족합니다."
        )


# ==================================================
# 17. 테스트 성능 평가
# ==================================================
st.divider()

st.subheader("🧪 최근 20년 테스트 성능")

st.markdown(
    """
    학습에 사용하지 않은 **2006~2025년 데이터를 테스트 데이터**로 사용했습니다.

    - **MAE:** 실제 기온과 예측 기온의 평균적인 차이
    - **MSE:** 오차를 제곱해 평균한 값
    - **R²:** 실제 기온의 변동을 모델이 얼마나 설명하는지 나타내는 값
    """
)


if len(테스트) >= 2:

    # ----------------------------------------------
    # 50년 모델
    # ----------------------------------------------
    st.markdown("### 🟦 최근 50년으로 학습한 모델")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "MAE",
            f"{mae50:.4f} ℃"
        )

    with col2:

        st.metric(
            "MSE",
            f"{mse50:.4f}"
        )

    with col3:

        st.metric(
            "R²",
            f"{r2_50:.4f}"
        )


    # ----------------------------------------------
    # 100년 모델
    # ----------------------------------------------
    st.markdown("### 🟩 최근 100년으로 학습한 모델")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "MAE",
            f"{mae100:.4f} ℃"
        )

    with col2:

        st.metric(
            "MSE",
            f"{mse100:.4f}"
        )

    with col3:

        st.metric(
            "R²",
            f"{r2_100:.4f}"
        )


    # ==================================================
    # 18. 성능 비교
    # ==================================================
    st.subheader("🏆 두 모델의 예측 성능 비교")

    비교_df = pd.DataFrame(
        {
            "모델": [
                "최근 50년 학습",
                "최근 100년 학습"
            ],
            "MAE(℃)": [
                mae50,
                mae100
            ],
            "MSE": [
                mse50,
                mse100
            ],
            "R²": [
                r2_50,
                r2_100
            ],
            "100년당 기온 변화(℃)": [
                기울기50 * 100,
                기울기100 * 100
            ]
        }
    )

    st.dataframe(
        비교_df.round(4),
        use_container_width=True,
        hide_index=True
    )


    # ==================================================
    # 19. 어느 모델이 더 좋은가
    # ==================================================
    mae_승자 = "최근 50년 학습" if mae50 < mae100 else "최근 100년 학습"
    mse_승자 = "최근 50년 학습" if mse50 < mse100 else "최근 100년 학습"
    r2_승자 = "최근 50년 학습" if r2_50 > r2_100 else "최근 100년 학습"

    st.info(
        f"""
        **평가 결과**

        - MAE가 더 작은 모델: **{mae_승자}**
        - MSE가 더 작은 모델: **{mse_승자}**
        - R²가 더 큰 모델: **{r2_승자}**

        일반적으로 **MAE와 MSE는 작을수록**, **R²는 클수록**
        테스트 데이터에 대한 예측 성능이 좋다고 볼 수 있습니다.
        """
    )


# ==================================================
# 20. 테스트 데이터 실제값 vs 예측값 그래프
# ==================================================
st.divider()

st.subheader("📊 테스트 데이터에서 실제 기온과 예측 기온 비교")

if len(테스트) >= 2:

    fig_test = go.Figure()

    # 실제값
    fig_test.add_trace(
        go.Scatter(
            x=테스트연도,
            y=테스트실제기온,
            mode="lines+markers",
            name="실제 연평균기온",
            line=dict(
                width=3
            ),
            marker=dict(
                size=7
            ),
            hovertemplate=(
                "연도: %{x}년<br>"
                "실제 기온: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )

    # 50년 모델
    fig_test.add_trace(
        go.Scatter(
            x=테스트연도,
            y=테스트예측50,
            mode="lines",
            name="50년 학습 모델 예측",
            line=dict(
                width=3,
                dash="dash"
            ),
            hovertemplate=(
                "연도: %{x}년<br>"
                "50년 모델 예측: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )

    # 100년 모델
    fig_test.add_trace(
        go.Scatter(
            x=테스트연도,
            y=테스트예측100,
            mode="lines",
            name="100년 학습 모델 예측",
            line=dict(
                width=3,
                dash="dot"
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
            "text": "2006~2025년 실제 기온과 두 회귀 모델의 예측 비교",
            "x": 0.5
        },
        xaxis=dict(
            title="연도",
            dtick=2,
            showgrid=True
        ),
        yaxis=dict(
            title="연평균기온(℃)",
            showgrid=True
        ),
        hovermode="x unified",
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
        fig_test,
        use_container_width=True
    )

else:

    st.warning(
        "테스트 데이터가 부족하여 그래프를 표시할 수 없습니다."
    )


# ==================================================
# 21. 학습 데이터와 회귀선 비교
# ==================================================
st.divider()

st.subheader("📈 50년 학습 vs 100년 학습 회귀선")

fig_train = go.Figure()

# 전체 실제 데이터
fig_train.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=6,
            opacity=0.55
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 50년 회귀선
if not np.isnan(기울기50):

    line50_years = np.linspace(
        학습50_시작,
        테스트_끝,
        200
    )

    line50_temp = 예측하기(
        line50_years,
        기울기50,
        절편50
    )

    fig_train.add_trace(
        go.Scatter(
            x=line50_years,
            y=line50_temp,
            mode="lines",
            name="50년 학습 회귀선",
            line=dict(
                width=3
            ),
            hovertemplate=(
                "연도: %{x:.0f}년<br>"
                "50년 모델: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )


# 100년 회귀선
if not np.isnan(기울기100):

    line100_years = np.linspace(
        학습100_시작,
        테스트_끝,
        300
    )

    line100_temp = 예측하기(
        line100_years,
        기울기100,
        절편100
    )

    fig_train.add_trace(
        go.Scatter(
            x=line100_years,
            y=line100_temp,
            mode="lines",
            name="100년 학습 회귀선",
            line=dict(
                width=3,
                dash="dash"
            ),
            hovertemplate=(
                "연도: %{x:.0f}년<br>"
                "100년 모델: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )


# 테스트 시작 / 종료 구간 표시
fig_train.add_vline(
    x=테스트_시작,
    line_width=2,
    line_dash="dot",
    annotation_text="테스트 시작"
)

fig_train.update_layout(
    height=600,
    template="plotly_white",
    title={
        "text": "서울 연평균기온과 50년·100년 학습 회귀선",
        "x": 0.5
    },
    xaxis=dict(
        title="연도",
        range=[1900, 2030],
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
    fig_train,
    use_container_width=True
)


# ==================================================
# 22. 전체 기간 회귀 분석
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

    전체 기간 회귀식:
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
# 23. 미래 기온 예측
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

predicted_temp = predict_temperature(
    selected_year
)

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
# 24. 기존 전체 기간 + 최근 20년 회귀선
# ==================================================
st.divider()

st.subheader("📉 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 관측값
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7,
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


# 전체 기간 회귀선
line_years = np.arange(
    1900,
    2101
)

전체회귀선 = (
    전체기울기
    * (line_years - 회귀기준연도)
    + 전체절편
)

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=전체회귀선,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "전체 기간 회귀 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 최근 20년 회귀선
if 최근20년_연도수 >= 2:

    최근선_연도 = np.linspace(
        최근시작연도,
        최근끝연도,
        100
    )

    최근회귀선 = (
        최근기울기
        * (최근선_연도 - 회귀기준연도)
        + 최근절편
    )

    fig.add_trace(
        go.Scatter(
            x=최근선_연도,
            y=최근회귀선,
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(
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


# 선택한 연도 예측값
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name="선택한 연도 예상 기온",
        marker=dict(
            size=14,
            line=dict(
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
# 25. 테스트 데이터 상세 표
# ==================================================
st.divider()

st.subheader("🧪 2006~2025년 테스트 데이터")

if len(테스트) >= 2:

    테스트표 = 테스트[
        ["연도", "연평균기온", "관측일수"]
    ].copy()

    테스트표["50년 모델 예측"] = 테스트예측50
    테스트표["100년 모델 예측"] = 테스트예측100

    테스트표["50년 모델 오차"] = (
        테스트표["연평균기온"]
        - 테스트표["50년 모델 예측"]
    )

    테스트표["100년 모델 오차"] = (
        테스트표["연평균기온"]
        - 테스트표["100년 모델 예측"]
    )

    테스트표["연평균기온"] = (
        테스트표["연평균기온"].round(2)
    )

    테스트표["50년 모델 예측"] = (
        테스트표["50년 모델 예측"].round(2)
    )

    테스트표["100년 모델 예측"] = (
        테스트표["100년 모델 예측"].round(2)
    )

    테스트표["50년 모델 오차"] = (
        테스트표["50년 모델 오차"].round(2)
    )

    테스트표["100년 모델 오차"] = (
        테스트표["100년 모델 오차"].round(2)
    )

    테스트표.columns = [
        "연도",
        "실제 연평균기온(℃)",
        "관측일수",
        "50년 모델 예측(℃)",
        "100년 모델 예측(℃)",
        "50년 모델 오차(℃)",
        "100년 모델 오차(℃)"
    ]

    st.dataframe(
        테스트표,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# 26. 전체 데이터 표
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
