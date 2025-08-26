import os
import json
from datetime import date, timedelta
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="버스 재배치 대시보드", layout="wide")
st.title("버스 재배치 대시보드")

# =========================
# 환경설정
# =========================
# API 베이스 URL: 환경변수 or 사이드바 입력
DEFAULT_BASE_URL = os.getenv("BUS_API_BASE_URL", "http://localhost:8000")
st.sidebar.subheader("데이터 소스")
data_source = st.sidebar.radio("가져올 소스 선택", ["CSV(로컬)", "API(HTTP)"], index=0)
base_url = st.sidebar.text_input("API Base URL", value=DEFAULT_BASE_URL, help="예: https://api.example.com")

route_no = st.sidebar.text_input("노선번호", value="172")
sel_date = st.sidebar.date_input("기준 날짜", value=date(2025, 6, 23))

# CSV 경로(기존 파이프라인 산출물)
CSV_PLAN = "temp/원정류장기준_버스재배치_계획.csv"
CSV_SCHED = "temp/원정류장기준_재배치후_출발스케줄.csv"

# =========================
# 헬퍼: API 호출
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def api_get(url: str, params: dict | None = None):
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API 호출 실패: {url}\n{e}")
        return None

def get_summary_api(route: str, d: date):
    url = f"{base_url}/v1/routes/{route}/days/{d.strftime('%Y-%m-%d')}/summary"
    return api_get(url)

def get_stop_detail_api(route: str, d: date, stop_seq: int):
    url = f"{base_url}/v1/routes/{route}/days/{d.strftime('%Y-%m-%d')}/stops/{stop_seq}"
    return api_get(url)

def get_forecast_api(route: str, start_d: date, days: int):
    url = f"{base_url}/v1/routes/{route}/forecast"
    return api_get(url, params={"start": start_d.strftime('%Y-%m-%d'), "days": days})

# =========================
# 데이터 적재 (CSV or API)
# =========================
if data_source == "CSV(로컬)":
    # 기존 CSV 로드
    try:
        plan_df = pd.read_csv(CSV_PLAN)
        sched_df = pd.read_csv(CSV_SCHED)
    except Exception as e:
        st.error(f"CSV 로드 실패: {e}")
        st.stop()

    # 날짜 필터
    plan_df["기준_날짜"] = pd.to_datetime(plan_df["기준_날짜"]).dt.strftime("%Y-%m-%d")
    sched_df["기준_날짜"] = pd.to_datetime(sched_df["기준_날짜"]).dt.strftime("%Y-%m-%d")
    sel_date_str = sel_date.strftime("%Y-%m-%d")

    filtered_plan = plan_df[plan_df["기준_날짜"] == sel_date_str]
    filtered_sched = sched_df[sched_df["기준_날짜"] == sel_date_str]

    # --- 재배치 계획 (자연어) ---
    st.subheader("재배치 계획 (CSV)")
    if filtered_plan.empty:
        st.info("선택한 날짜의 재배치 계획이 없습니다.")
    else:
        for _, row in filtered_plan.iterrows():
            st.write(f"🚍 {int(row['from_hour'])}시 → {int(row['to_hour'])}시로 1대 이동 (bus_id={row['move_bus_id']})")

    # --- 재배치 후 출발 스케줄 (표) ---
    st.subheader("재배치 후 출발 스케줄 (CSV)")
    if filtered_sched.empty:
        st.info("선택한 날짜의 출발 스케줄이 없습니다.")
    else:
        table = filtered_sched.rename(
            columns={"도착_시간": "도착 시간", "slot_idx": "슬롯 번호", "균등_출발시각": "균등 출발 시각"}
        ).copy()
        # 보기 좋게 정렬
        table = table.sort_values(["도착 시간", "슬롯 번호"]).reset_index(drop=True)
        st.dataframe(table, use_container_width=True, hide_index=True)

else:
    # =========================
    # API 모드
    # =========================
    st.caption("모드: API(HTTP)")

    # 1) 일자별 시간대 계획 요약
    st.subheader("일자별 시간대 계획 요약")
    summary = get_summary_api(route_no, sel_date)
    if summary is None:
        st.stop()

    # summary 렌더
    try:
        hours = pd.DataFrame(summary.get("hours", []))
        if not hours.empty:
            # 자연어 라인: 증차/감차만 강조
            change_lines = []
            for _, r in hours.iterrows():
                cur = int(r.get("current_departures", 0))
                rec = int(r.get("recommended_departures", cur))
                hour = int(r.get("hour", 0))
                if rec > cur:
                    change_lines.append(f"🔺 {hour}시: 현재 {cur}대 → 권장 {rec}대 (헤드웨이 ≈ {r.get('headway_min','-')}분, {r.get('congestion','-')})")
                elif rec < cur:
                    change_lines.append(f"🔻 {hour}시: 현재 {cur}대 → 권장 {rec}대 (헤드웨이 ≈ {r.get('headway_min','-')}분, {r.get('congestion','-')})")
            if change_lines:
                st.markdown("\n".join(f"- {line}" for line in change_lines))
            else:
                st.info("이 날짜는 권장 변경 사항이 없습니다.")

            st.dataframe(
                hours.rename(columns={
                    "hour":"시간",
                    "current_departures":"현재 출발대수",
                    "recommended_departures":"권장 출발대수",
                    "headway_min":"헤드웨이(분)",
                    "congestion":"혼잡도",
                    "pivot_stop_seq":"피크 기준 정류장 순서",
                    "pivot_stop_name":"피크 기준 정류장명"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("요약 데이터가 비어 있습니다.")
    except Exception as e:
        st.error(f"요약 렌더 실패: {e}")

    # 2) 정류장 상세
    st.subheader("정류장 상세")
    # stop_seq 선택(요약의 pivot_stop_seq 우선 사용, 없으면 입력)
    default_stop_seq = None
    if summary and summary.get("hours"):
        # 가장 혼잡 시간대의 pivot_stop_seq
        try:
            hdf = pd.DataFrame(summary["hours"])
            hdf["rank"] = hdf.get("recommended_departures", 0) - hdf.get("current_departures", 0)
            hdf = hdf.sort_values("rank", ascending=False)
            default_stop_seq = int(hdf.iloc[0].get("pivot_stop_seq")) if "pivot_stop_seq" in hdf.columns else None
        except Exception:
            default_stop_seq = None
    stop_seq = st.number_input("정류장 순서(stop_seq)", min_value=1, value=default_stop_seq or 1, step=1)

    stop_detail = get_stop_detail_api(route_no, sel_date, int(stop_seq))
    if stop_detail and stop_detail.get("hours"):
        st.dataframe(
            pd.DataFrame(stop_detail["hours"]).rename(columns={
                "hour":"시간",
                "buses":"통과 버스수",
                "pred_board_total":"예측 승차(총)",
                "pred_alight_total":"예측 하차(총)",
                "per_bus_onboard":"버스당 탑승",
                "congestion":"혼잡도"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("정류장 상세 데이터가 없습니다.")

    # 3) 7일 예측 조회
    st.subheader("7일 예측 조회")
    days = st.slider("조회 일수", min_value=3, max_value=14, value=7, step=1)
    forecast = get_forecast_api(route_no, sel_date, days)
    if forecast and forecast.get("days"):
        # 간단한 테이블로 요약
        rows = []
        for d in forecast["days"]:
            for h in d.get("hours", []):
                rows.append({
                    "날짜": d.get("date"),
                    "시간": h.get("hour"),
                    "현재 출발": h.get("current_departures"),
                    "권장 출발": h.get("recommended_departures"),
                    "혼잡도": h.get("congestion")
                })
        fdf = pd.DataFrame(rows)
        st.dataframe(fdf, use_container_width=True, hide_index=True)
    else:
        st.info("예측 데이터가 없습니다.")

st.markdown("---")
st.caption("데이터 소스 전환: 사이드바에서 CSV ↔ API 선택 가능. API 오류 시 CSV로 작업을 이어가세요.")
