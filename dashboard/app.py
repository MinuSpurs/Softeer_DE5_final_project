import streamlit as st
import pandas as pd
import plotly.express as px

st.title("버스 재배치 대시보드")

# Load data
plan_df = pd.read_csv("temp/원정류장기준_버스재배치_계획.csv")
schedule_df = pd.read_csv("temp/원정류장기준_재배치후_출발스케줄.csv")

# Sidebar date selection
dates = plan_df['기준_날짜'].unique()
selected_date = st.sidebar.selectbox("기준_날짜 선택", dates)

# Filter data by selected date
filtered_plan = plan_df[plan_df['기준_날짜'] == selected_date]
filtered_schedule = schedule_df[schedule_df['기준_날짜'] == selected_date]

st.header("재배치 계획")
# Display human-readable sentences for each row in filtered_plan
for idx, row in filtered_plan.iterrows():
    st.write(f"🚍 {row['from_hour']}시 버스를 {row['to_hour']}시 버스로 옮겨야 합니다. (move_bus_id={row['move_bus_id']})")

# --- New bar chart and heatmap based on from_hour and to_hour ---
# Bar chart: count of reallocations per from_hour
hour_counts = filtered_plan['from_hour'].value_counts().sort_index().reset_index()
hour_counts.columns = ['from_hour', '재배치 횟수']
fig_hour_bar = px.bar(hour_counts, x='from_hour', y='재배치 횟수', title="from_hour별 재배치 횟수")
st.plotly_chart(fig_hour_bar)

# Heatmap: from_hour vs to_hour
heatmap_hours = filtered_plan.groupby(['from_hour', 'to_hour']).size().reset_index(name='count')
pivot_hour_heatmap = heatmap_hours.pivot(index='from_hour', columns='to_hour', values='count').fillna(0)
fig_hour_heatmap = px.imshow(
    pivot_hour_heatmap,
    labels=dict(x="to_hour", y="from_hour", color="재배치 횟수"),
    x=pivot_hour_heatmap.columns,
    y=pivot_hour_heatmap.index,
    aspect="auto",
    title="from_hour → to_hour 재배치 Heatmap"
)
st.plotly_chart(fig_hour_heatmap)
# --- End new code ---

st.header("재배치 후 출발 스케줄")
st.dataframe(filtered_schedule)

# Timeline scatter plot
# x-axis: 균등_출발시각, y-axis: 정류장 순서, color: slot_idx
if {'균등_출발시각', '정류장 순서', 'slot_idx'}.issubset(filtered_schedule.columns):
    fig_timeline = px.scatter(filtered_schedule,
                              x='균등_출발시각',
                              y='정류장 순서',
                              color='slot_idx',
                              title="재배치 후 출발 스케줄 타임라인",
                              labels={'균등_출발시각': '균등 출발 시각', '정류장 순서': '정류장 순서', 'slot_idx': 'Slot Index'},
                              hover_data=filtered_schedule.columns)
    st.plotly_chart(fig_timeline)
