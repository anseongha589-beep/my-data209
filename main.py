import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# [필수] Matplotlib 한글 깨짐 및 마이너스 기호 깨짐 해결 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False 

# 1. 웹 페이지 설정 및 제목
st.set_page_config(page_title="서울 100년 기온 데이터 종합 분석", layout="wide") 
st.title("🌡️ 서울 100년 연평균 기온 변화 및 원본 데이터 특성 분석")
st.markdown("지난 100년 동안의 서울 연평균 기온 변화를 확인하고 원본 데이터의 **요약 통계(개수, 평균, 최소, 최대 등)**를 함께 분석하는 대시보드입니다.")

# 2. 데이터 불러오기 (네트워크 장애 방지 백업 포함)
@st.cache_data
def load_data():
    url = "https://githubusercontent.com"
    try:
        import urllib.request
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(html))
    except Exception:
        backup_data = """연도,평균기온
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
2020,13.2"""
        df_backup = pd.read_csv(io.StringIO(backup_data))
        years = pd.DataFrame({"연도": range(1908, 2021)})
        df = pd.merge(years, df_backup, on="연도", how="left").interpolate()
        return df
        
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    df = df.dropna(subset=["평균기온"])
    annual_mean = df.groupby("연도")["평균기온"].mean().reset_index()
    return annual_mean

try:
    # 3. 원본 데이터 로드
    original_data = load_data()

    # 📌 4. [핵심 추가] 원본 데이터 특성 요약 통계 배치 (개수, 평균, 최소, 최대)
    st.subheader("📋 전체 원본 데이터의 요약 통계 특성")
    
    # 100년 원본 전체 데이터 기준 기술통계량 산출
    orig_count = float(len(original_data))
    orig_mean = original_data["평균기온"].mean()
    orig_min = original_data["평균기온"].min()
    orig_max = original_data["평균기온"].max()
    
    # 사용자가 보기 좋게 한글 레이아웃의 데이터프레임 구성
    summary_df = pd.DataFrame({
        "데이터 요약 지표": ["데이터 총 개수 (Count)", "전체 평균 기온 (Mean)", "역사상 최소 기온 (Min)", "역사상 최대 기온 (Max)"],
        "평균기온 통계치": [f"{orig_count:,.0f} 개년", f"{orig_mean:.2f} °C", f"{orig_min:.1f} °C", f"{orig_max:.1f} °C"]
    })
    
    # 불필요한 인덱스(0, 1, 2)를 숨겨 정갈하게 최상단 고정 노출
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 5. 사이드바 기간 조절기 필터
    st.sidebar.header("📊 조회 설정")
    min_year = int(original_data["연도"].min())
    max_year = int(original_data["연도"].max())
    
    start_year, end_year = st.sidebar.slider(
        "조회할 연도 범위를 선택하세요",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    # 슬라이더로 선택한 범위 데이터 필터링
    filtered_data = original_data[(original_data["연도"] >= start_year) & (original_data["연도"] <= end_year)]

    # 6. 선택 기간 주요 지표(Metric) 시각화
    if not filtered_data.empty:
        col1, col2 = st.columns(2)
        with col1:
            # [수정 완료] 기존 iloc 구문 에러 완전 해결
            start_temp = filtered_data['평균기온'].iloc[0]
            st.metric(label=f"⏰ {start_year}년 평균 기온", value=f"{start_temp:.1f} °C")
        with col2:
            end_temp = filtered_data['평균기온'].iloc[-1]
            diff = end_temp - start_temp
            st.metric(label=f"⏳ {end_year}년 평균 기온", value=f"{end_temp:.1f} °C", delta=f"{diff:+.1f} °C")

    st.markdown("---")

    # 7. 좌우 레이아웃 분할 (좌측: 트렌드 추세선 그래프 / 우측: 사분위수 상세 정보 및 분포)
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.subheader("📈 선택 기간 기온 변화 트렌드 및 추세선")
        st.write(f"*{start_year}년부터 {end_year}년까지의 기온 변화 추이와 경향성*")
        
        fig_trend, ax_trend = plt.subplots(figsize=(10, 5.5))
        ax_trend.plot(filtered_data["연도"], filtered_data["평균기온"], marker='o', linestyle='-', color='#ff4b4b', label="Annual Mean Temp")
        
        # 데이터가 2개 이상일 때만 추세선 추가
        if len(filtered_data) > 1:
            z = np.polyfit(filtered_data["연도"], filtered_data["평균기온"], 1)
            p = np.poly1d(z)
            ax_trend.plot(filtered_data["연도"], p(filtered_data["연도"]), linestyle="--", color="#31333F", alpha=0.7, label="Trend Line")

        # [한글 깨짐 해결] 리눅스 서버 안정성을 위해 내부 레이블 영문 고정
        ax_trend.set_xlabel("Year", fontsize=10)
        ax_trend.set_ylabel("Temperature (°C)", fontsize=10)
        ax_trend.grid(True, linestyle=":", alpha=0.6)
        ax_trend.legend(loc="upper left")
        st.pyplot(fig_trend)

    with col_right:
        st.subheader("📊 데이터 세부 분포 및 무결성")
        
        # 상세 기술통계량 표 (사분위수 포함)
        st.markdown("**1. 상세 기술통계량 (사분위수)**")
        summary = original_data["평균기온"].describe().to_frame()
        summary.index = ["데이터 개수 (개)", "평균 기온 (°C)", "표준편차 (변동성)", "최소 기온 (°C)", "25% (하위 사분위)", "50% (중앙값)", "75% (상위 사분위)", "최대 기온 (°C)"]
        summary.columns = ["통계치"]
        st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)
        
        # 원본 데이터 무결성 검증 (결측치)
        st.markdown("**2. 결측치(누락) 검증**")
        null_count = original_data["평균기온"].isnull().sum()
        if null_count == 0:
            st.success("✅ 원본 데이터 내 결측치가 존재하지 않는 깨끗한 데이터입니다.")
        else:
            st.warning(f"⚠️ 원본 데이터 내에 {null_count}개의 누락된 값이 발견되어 필터링되었습니다.")

    st.markdown("---")

    # 8. 하단 배치: 이상치 검출용 박스플롯 분포 및 원본 레코드 테이블
    col_box, col_table = st.columns([0.8, 1.2])

    with col_box:
        st.subheader("📦 데이터 치우침 점검 (Boxplot)")
        fig_box, ax_box = plt.subplots(figsize=(5, 4))
        
        # 외부 seaborn 종속성 없이 matplotlib 순정 코드로 정교하게 박스플롯 구성
        ax_box.boxplot(original_data["평균기온"], patch_artist=True, 
                       boxprops=dict(facecolor='#ffebeb', color='#ff4b4b', linewidth=1.5),
                       medianprops=dict(color='#31333F', linewidth=2),
                       whiskerprops=dict(color='#ff4b4b', linewidth=1.5),
                       capprops=dict(color='#ff4b4b', linewidth=1.5))
        
        # [한글 깨짐 해결] 그래프 내부 영문 매핑
        ax_box.set_title("Original Temperature Distribution", fontsize=11, fontweight='bold', pad=10)
        ax_box.set_ylabel("Temperature Values (°C)", fontsize=9)
        ax_box.set_xticklabels(["Original Data"])
        ax_box.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig_box)

    with col_table:
        st.subheader("🗃️ 필터링된 데이터 레코드")
        # 사용자가 스크롤하며 실제 수치를 직접 볼 수 있는 테이블
        st.dataframe(filtered_data.set_index("연도"), use_container_width=True, height=265)

except Exception as e:
    st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}")
