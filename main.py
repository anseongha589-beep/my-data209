import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 웹 페이지 설정 및 제목
st.set_page_config(page_title="서울 100년 기온 변화", layout="centered")
st.title("🌡️ 서울 100년 연평균 기온 변화 추이")
st.markdown("지난 100년 동안 서울의 연평균 기온이 어떻게 변해왔는지 확인하는 대시보드입니다.")

# 2. 데이터 불러오기 (캐싱 처리로 속도 향상)
@st.cache_data
def load_data():
    url = "https://githubusercontent.com"
    
    # 인코딩 오류 해결을 위해 'utf-8'로 변경하여 로드
    df = pd.read_csv(url, encoding="utf-8")
    
    # '날짜' 열을 데이트타임 형태로 변환
    df["날짜"] = pd.to_datetime(df["날짜"])
    # 연도 추출
    df["연도"] = df["날짜"].dt.year
    
    # 연도별 평균 기온 계산
    annual_mean = df.groupby("연度")["평균기온"].mean().reset_index() if "연度" in df.columns else df.groupby("연도")["평균기온"].mean().reset_index()
    return annual_mean

try:
    data = load_data()

    # 3. 사이드바 제어 조절기 (연도 선택 사이드바)
    st.sidebar.header("📊 조회 설정")
    min_year = int(data["연도"].min())
    max_year = int(data["연도"].max())
    
    selected_years = st.sidebar.slider(
        "조회할 연도 범위를 선택하세요",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    # 선택한 연도 데이터 필터링 (오타 수정 완료)
    filtered_data = data[(data["연도"] >= selected_years[0]) & (data["연도"] <= selected_years[1])]

    # 4. 주요 지표(Metric) 시각화
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"{selected_years[0]}년 평균 기온", value=f"{filtered_data['평균기온'].iloc[0]:.1f} °C")
    with col2:
        st.metric(label=f"{selected_years[1]}년 평균 기온", value=f"{filtered_data['평균기온'].iloc[-1]:.1f} °C")

    # 5. 그래프 그리기
    st.subheader("📈 연평균 기온 변화 그래프")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(filtered_data["연도"], filtered_data["평균기온"], marker='o', linestyle='-', color='#ff4b4b', label="연평균 기온")
    
    # 추세선 추가
    z = np.polyfit(filtered_data["연도"], filtered_data["평균기온"], 1)
    p = np.poly1d(z)
    ax.plot(filtered_data["연도"], p(filtered_data["연도"]), linestyle="--", color="#31333F", alpha=0.7, label="기온 추세선")

    # 그래프 스타일링
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    
    st.pyplot(fig)

    # 6. 데이터 표 확인
    if st.checkbox("전체 데이터 테이블 보기"):
        st.dataframe(filtered_data.set_index("연도"))

except Exception as e:
    st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}")
