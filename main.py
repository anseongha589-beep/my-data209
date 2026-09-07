    # =====================================================
    # 오른쪽 : 원본 데이터 분포
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
        # 분포 핵심 수치
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


        # 평균 위치 표시
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
        # 분포 해석
        # -------------------------------------------------
        st.markdown(
            f"""
            <div class="tip-box">
            <b>📌 분포 요약</b><br><br>
            • 평균기온: <b>{orig_mean:.2f} °C</b><br>
            • 중앙값: <b>{orig_median:.2f} °C</b><br>
            • 50%의 데이터가 포함된 범위:
              <b>{orig_q1:.2f} ~ {orig_q3:.2f} °C</b><br>
            • 전체 기온 범위:
              <b>{orig_min:.1f} ~ {orig_max:.1f} °C</b><br>
            • 사분위간 범위(IQR):
              <b>{orig_iqr:.2f} °C</b>
            </div>
            """,
            unsafe_allow_html=True
        )
