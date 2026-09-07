```python
with col_left:
    st.subheader("📈 선택 기간 기온 변화 트렌드 및 추세선")
    st.write(f"*{start_year}년부터 {end_year}년까지의 기온 변화 추이와 경향성*")

    # Plotly 인터랙티브 그래프 생성
    fig_trend = go.Figure()

    # 연평균 기온 그래프
    fig_trend.add_trace(
        go.Scatter(
            x=filtered_data["연도"],
            y=filtered_data["평균기온"],
            mode="lines+markers",
            name="연평균 기온",
            line=dict(color="#ff4b4b", width=2),
            marker=dict(size=7),
            customdata=filtered_data[["연도"]],
            hovertemplate=
                "<b>%{customdata[0]}년</b><br>" +
                "평균기온: %{y:.1f} °C" +
                "<extra></extra>"
        )
    )

    # 추세선
    if len(filtered_data) > 1:
        z = np.polyfit(
            filtered_data["연도"],
            filtered_data["평균기온"],
            1
        )
        p = np.poly1d(z)

        fig_trend.add_trace(
            go.Scatter(
                x=filtered_data["연도"],
                y=p(filtered_data["연도"]),
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

    fig_trend.update_layout(
        xaxis_title="연도",
        yaxis_title="평균기온 (°C)",
        hovermode="closest",
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            x=0.01,
            y=0.99
        )
    )

    # Streamlit에 인터랙티브 그래프 표시
    st.plotly_chart(
        fig_trend,
        use_container_width=True
    )
```
