# /Users/minwoo/Desktop/softeer/DE5_Final_project/dashboard/app.py
import pandas as pd
import streamlit as st
from datetime import date, timedelta

st.set_page_config(page_title="버스 재배치 대시보드", layout="wide")
st.title("버스 재배치 대시보드")

# =========================
# CSV 경로 (기존 산출물)
# =========================
CSV_PLAN  = "temp/원정류장기준_버스재배치_계획.csv"      # 기준_날짜, move_bus_id, from_hour, to_hour
CSV_SCHED = "temp/원정류장기준_재배치후_출발스케줄.csv"  # 기준_날짜, 도착_시간, slot_idx, 균등_출발시각

@st.cache_data(ttl=300, show_spinner=False)
def load_csv(path): return pd.read_csv(path)

# 로드 & 날짜 문자열 정규화
try:
    plan_df = load_csv(CSV_PLAN)
    sched_df = load_csv(CSV_SCHED)
except Exception as e:
    st.error(f"CSV 로드 실패: {e}")
    st.stop()

for df, col in ((plan_df, "기준_날짜"), (sched_df, "기준_날짜")):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")

# 날짜 선택: 두 CSV의 날짜 union
all_days = sorted(
    set(plan_df.get("기준_날짜", pd.Series(dtype=str))).union(
        set(sched_df.get("기준_날짜", pd.Series(dtype=str)))
    )
)
default_day = all_days[0] if all_days else date(2025, 6, 23).strftime("%Y-%m-%d")
sel_day_str = st.sidebar.selectbox("기준 날짜", options=all_days or [default_day], index=0)

st.markdown("---")
tab_sum, tab_stop, tab_fore = st.tabs([
    "📊 [읽기] 일자별 시간대 계획 요약",
    "🧭 [읽기] 정류장 상세",
    "📅 [읽기] 7일 예측 조회"
])

# 공통 유틸
def _safe_int(x):
    try: return int(x)
    except: return x

# =========================
# 1) [읽기] 일자별 시간대 계획 요약
# =========================
with tab_sum:
    st.subheader(f"[일자별 시간대 계획 요약] — {sel_day_str}")
    pday = plan_df[plan_df["기준_날짜"] == sel_day_str].copy() if "기준_날짜" in plan_df.columns else pd.DataFrame()

    if pday.empty:
        st.info("선택한 날짜의 재배치 계획이 없습니다.")
    else:
        # 시간대별 이동 out/in/net 요약
        moved_out = pday.groupby("from_hour").size().reset_index(name="moved_out")
        moved_in  = pday.groupby("to_hour").size().reset_index(name="moved_in")

        # 0~23 전 시간대 프레임
        hours = pd.DataFrame({"hour": list(range(24))})
        hours = (hours.merge(moved_out.rename(columns={"from_hour":"hour"}), on="hour", how="left")
                      .merge(moved_in.rename(columns={"to_hour":"hour"}),  on="hour", how="left"))
        hours["moved_out"] = hours["moved_out"].fillna(0).astype(int)
        hours["moved_in"]  = hours["moved_in"].fillna(0).astype(int)
        hours["net_change"] = hours["moved_in"] - hours["moved_out"]

        # 자연어 요약(건별)
        st.write("#### 이동 내역")
        for _, r in pday.iterrows():
            st.write(f"🚍 {int(r['from_hour'])}시 → {int(r['to_hour'])}시로 1대 이동 (bus_id={r['move_bus_id']})")

        # 요약 표
        st.write("#### 시간대별 이동 요약 (유입/유출/순변화)")
        pretty = hours.rename(columns={
            "hour": "시간",
            "moved_out": "유출(해당 시간에서 나감)",
            "moved_in": "유입(해당 시간으로 들어옴)",
            "net_change": "순변화(유입-유출)"
        })
        st.dataframe(pretty, use_container_width=True, hide_index=True)

        # from_hour x to_hour 매트릭스(표)
        st.write("#### 시간대별 이동 매트릭스 (from → to, 건수)")
        mat = (pday.groupby(["from_hour","to_hour"])
                    .size().reset_index(name="count")
                    .pivot(index="from_hour", columns="to_hour", values="count")
                    .fillna(0).astype(int)
                    .rename_axis(index="from_hour", columns="to_hour")
              )
        st.dataframe(mat, use_container_width=True)

# =========================
# 2) [읽기] 정류장 상세
#     - 현재 CSV(스케줄)에는 정류장 ID/순서가 없음 → 시간대·슬롯 기준 상세 테이블 제공
# =========================
with tab_stop:
    st.subheader(f"[정류장 상세] — {sel_day_str}")
    sday = sched_df[sched_df["기준_날짜"] == sel_day_str].copy() if "기준_날짜" in sched_df.columns else pd.DataFrame()

    if sday.empty:
        st.info("선택한 날짜의 스케줄 데이터가 없습니다.")
    else:
        # 도착_시간별 슬롯 개수 요약
        if "도착_시간" in sday.columns and "slot_idx" in sday.columns:
            agg = (sday.groupby("도착_시간")["slot_idx"]
                        .nunique().reset_index(name="슬롯_개수(해당 시에 출발 예정 대수)"))
            agg = agg.sort_values("도착_시간")
            st.write("#### 시간대별 출발 슬롯 요약")
            st.dataframe(agg, use_container_width=True, hide_index=True)

        # 상세 테이블
        pretty = sday.rename(columns={
            "기준_날짜":"날짜",
            "도착_시간":"시간",
            "slot_idx":"슬롯 번호",
            "균등_출발시각":"균등 출발 시각"
        }).sort_values(["시간","슬롯 번호"]).reset_index(drop=True)

        st.write("#### 상세(시간·슬롯 순)")
        st.dataframe(pretty, use_container_width=True, hide_index=True)

        # 자연어 라인 (간단)
        st.write("#### 안내")
        for _, r in pretty.iterrows():
            st.write(f"🕒 {r['시간']}시 슬롯 {r['슬롯 번호']} — 균등 출발 {r['균등 출발 시각']}")

# =========================
# 3) [읽기] 7일 예측 조회 (계획 CSV 기반 집계)
#     - 별도 예측 CSV 없이, '계획 CSV'를 7일 윈도우로 요약해서 보여줌
# =========================
with tab_fore:
    st.subheader("[7일 예측 조회] (계획 집계 기반)")
    # 시작일 선택
    if all_days:
        start_day = st.date_input("시작일", value=pd.to_datetime(all_days[0]).date(), min_value=pd.to_datetime(all_days[0]).date())
    else:
        start_day = st.date_input("시작일", value=date(2025,6,23))

    days_n = st.number_input("조회 일수(N)", min_value=1, max_value=14, value=7, step=1)

    if plan_df.empty:
        st.info("계획 CSV가 비어 있습니다.")
    else:
        # 윈도우 필터
        s = pd.to_datetime(start_day).strftime("%Y-%m-%d")
        e = (pd.to_datetime(start_day) + timedelta(days=int(days_n)-1)).strftime("%Y-%m-%d")
        wnd = plan_df[(plan_df["기준_날짜"]>=s) & (plan_df["기준_날짜"]<=e)].copy()

        if wnd.empty:
            st.info(f"{s} ~ {e} 범위에 데이터가 없습니다.")
        else:
            st.write(f"기간: **{s} ~ {e}**")

            # 일자별 총 이동 건수
            daily = (wnd.groupby("기준_날짜").size()
                        .reset_index(name="총_재배치_건수")
                        .sort_values("기준_날짜"))
            st.write("#### 일자별 총 재배치 건수")
            st.dataframe(daily, use_container_width=True, hide_index=True)

            # 일자×시간 순변화
            out_d = wnd.groupby(["기준_날짜","from_hour"]).size().reset_index(name="out")
            in_d  = wnd.groupby(["기준_날짜","to_hour"]).size().reset_index(name="in")
            hours_d = (out_d.rename(columns={"from_hour":"시간"})
                           .merge(in_d.rename(columns={"to_hour":"시간"}), on=["기준_날짜","시간"], how="outer"))
            hours_d["out"] = hours_d["out"].fillna(0).astype(int)
            hours_d["in"]  = hours_d["in"].fillna(0).astype(int)
            hours_d["net_change"] = hours_d["in"] - hours_d["out"]
            hours_d = hours_d.sort_values(["기준_날짜","시간"]).reset_index(drop=True)

            st.write("#### 일자×시간 순변화(유입-유출)")
            pretty2 = hours_d.rename(columns={"시간":"hour"})
            st.dataframe(pretty2, use_container_width=True, hide_index=True)

            # 자연어 요약 몇 줄
            st.write("#### 요약(샘플 10건)")
            for _, r in wnd.head(10).iterrows():
                st.write(f"📌 {r['기준_날짜']}: {int(r['from_hour'])}시 → {int(r['to_hour'])}시로 1대 이동 (bus_id={r['move_bus_id']})")

st.markdown("---")
st.caption("모드: CSV(로컬) — 경로는 상단 CSV_* 상수를 수정하세요.")