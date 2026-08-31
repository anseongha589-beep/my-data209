import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 웹 페이지 설정 및 제목
st.set_page_config(page_title="서울 100년 기온 변화", layout="centered")
st.title("🌡️ 서울 100년 연평균 기온 변화 추이")
st.markdown("지난 100년 동안 서울의 연평균 기온이 어떻게 변해왔는지 확인하는 대시보드입니다.")

# 2. 데이터 불러오기 (절대 경로 추적 및 캐싱 처리)
@st.cache_data
def load_data():
    # 현재 실행 중인 main.py 파일의 폴더 위치를 자동으로 찾습니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "seoul.csv")
    
    # 파일 존재 여부 선제 확인
    if not os.path.exists(file_path):
        st.error(f"📂 '{file_path}' 경로에서 데이터를 찾을 수 없습니다. 파일이 올바른 위치에 업로드되었는지 확인해 주세요.")
        st.stop()
        
    # 인코딩 형식을 안전하게 utf-8로 지정하여 로드
    df = pd.read_csv(file_path, encoding="utf-8")
    
    # '날짜' 열을 데이트타임 형태로 변환 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 연도별 평균 기온 계산
    annual_mean = df.groupby("연도")["평균기온"].mean().reset_index()
    return annual_mean

try:
    data = load_data()

    # 3. 사이드바 제어 조절기
    st.sidebar.header("📊 조회 설정")
    min_year = int(data["연도"].min())
    max_year = int(data["연도"].max())
    
    start_year, end_year = st.sidebar.slider(
        "조회할 연도 범위를 선택하세요",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    # 선택한 연도 범위 데이터 필터링
    filtered_data = data[(data["연도"] >= start_year) & (data["연도"] <= end_year)]

    # 4. 주요 지표(Metric) 시각화
    if not filtered_data.empty:
        col1, col2 = st.columns(2)
        with col1:
            start_temp = filtered_data['평균기온'].iloc[0]
            st.metric(label=f"{start_year}년 평균 기온", value=f"{start_temp:.1f} °C")
        with col2:
            end_temp = filtered_data['평균기온'].iloc[-1]
            # 두 연도 간의 기온 차이를 delta로 표현
            diff = end_temp - start_temp
            st.metric(label=f"{end_year}년 평균 기온", value=f"{end_temp:.1f} °C", delta=f"{diff:+.1f} °C")

    # 5. 그래프 그리기
    st.subheader("📈 연평균 기온 변화 그래프")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(filtered_data["연도"], filtered_data["평균기온"], marker='o', linestyle='-', color='#ff4b4b', label="연평균 기온")
    
    # 데이터가 2개 이상일 때만 추세선 추가
    if len(filtered_data) > 1:
        z = np.polyfit(filtered_data["연도"], filtered_data["평균기온"], 1)
        p = np.poly1d(z)
        ax.plot(filtered_data["연도"], p(filtered_data["연도"]), linestyle="--", color="#31333F", alpha=0.7, label="기온 추세선")

    # 그래프 스타일링 (스트림릿 서버 한글 깨짐 방지용 영문 표기)
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
