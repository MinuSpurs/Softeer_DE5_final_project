import pandas as pd
import numpy as np
from math import ceil

# ---------- 노트북/스크립트 겸용 출력 ----------
def safe_display(df, head=30, title=None):
    try:
        from IPython.display import display
        if title:
            print(title)
        display(df.head(head) if hasattr(df, "head") else df)
    except Exception:
        if title:
            print(title)
        print(df.head(head).to_string(index=False) if hasattr(df, "head") else str(df))

# ---------- 요일그룹(주중/주말) ----------
def get_dow_group(weekday_idx: int) -> str:
    # weekday_idx: 0=월 ... 6=일
    return "주말" if weekday_idx >= 5 else "주중"

# ----------------------------------------------------
# 보호구간(첫차+3h, 막차-3h) 계산: 기점(정류장_순서==1) 기준
# ----------------------------------------------------
def compute_protected_windows(schedule, protect_h=3):
    # schedule: bus-level 또는 근사 스케줄, 최소 [기준_날짜, 도착_시간, 정류장_순서]
    first_last = (
        schedule[schedule["정류장_순서"]==1]
        .groupby("기준_날짜")["도착_시간"]
        .agg(first_hour="min", last_hour="max")
        .reset_index()
    )
    first_last["protect_start_hi"] = first_last["first_hour"] + protect_h
    first_last["protect_end_lo"]   = first_last["last_hour"]  - protect_h
    return first_last

# ----------------------------------------------------
# 혼잡 타깃/여유 공여 시간대 만들기
# hourly_need: [기준_날짜, to_hour(혼잡 시단), needed_before, pivot_stop_seq, pivot_stop_name]
# hourly_supply: [기준_날짜, from_hour(여유 시단), spare(=뺄 수 있는 대수, 정수)]
# ----------------------------------------------------
def build_targets_and_donors(hourly_need, hourly_supply, schedule, protect_h=3, debug=True):
    fl = compute_protected_windows(schedule, protect_h=protect_h)
    # 타깃: 혼잡 시단 (needed_before>0) + 보호구간 바깥
    tgt = (hourly_need.copy()
           .rename(columns={"시간":"to_hour"})
           )
    tgt = tgt.merge(fl, on="기준_날짜", how="left")
    tgt = tgt[(tgt["needed_before"]>0) &
              (tgt["to_hour"]>=tgt["protect_start_hi"]) &
              (tgt["to_hour"]<=tgt["protect_end_lo"])].copy()

    # 도너: 여유 시단 (spare>0) + 보호구간 바깥
    don = (hourly_supply.copy()
           .rename(columns={"시간":"from_hour"})
           )
    don = don.merge(fl, on="기준_날짜", how="left")
    don = don[(don["spare"]>0) &
              (don["from_hour"]>=don["protect_start_hi"]) &
              (don["from_hour"]<=don["protect_end_lo"])].copy()

    if debug:
        dbg = (f"[DEBUG] 타깃(혼잡) 시단 수: {len(tgt)} / "
               f"도너(여유) 시단 수: {len(don)}\n"
               f"보호구간(first+{protect_h}, last-{protect_h}) 적용 후 남은 시단만 대상으로 합니다.")
        print(dbg)
        if len(tgt)==0:
            print(" - 혼잡 타깃이 모두 보호구간에 있거나 이미 해소되어 제안이 없습니다.")
        if len(don)==0:
            print(" - 여유 도너가 보호구간에 있거나 spare=0이라 제안이 제한됩니다.")

    return tgt, don

# ----------------------------------------------------
# τ lookup 이용해서 출발시각 산정 + 그리디 매칭 (상/하행 구분 없이)
# hybrid_tau: [정류장_순서, 도착_시간, 요일그룹, tau_final_min, tau_source]
# get_dow_group: 요일 그룹핑 함수(주중/주말 등)
# ----------------------------------------------------
def plan_reallocation(tgt, don, hybrid_tau, get_dow_group,
                      headway_min=10, max_moves_per_target=99, debug=True):
    plans = []

    # 도너는 spare 큰 순, 타깃은 needed 큰 순으로 우선 매칭
    tgt = tgt.sort_values(["기준_날짜","needed_before"], ascending=[True, False]).reset_index(drop=True)
    don = don.sort_values(["기준_날짜","spare"], ascending=[True, False]).reset_index(drop=True)

    for day, tgt_day in tgt.groupby("기준_날짜"):
        don_day = don[don["기준_날짜"]==day].copy()
        if don_day.empty:
            if debug: print(f"[DEBUG] {day} 도너 없음 → 스킵")
            continue

        for _, row in tgt_day.iterrows():
            need = int(row["needed_before"])
            if need<=0:
                continue

            to_hour   = int(row["to_hour"])
            pivot_seq = int(row["pivot_stop_seq"])
            pivot_name= row.get("pivot_stop_name","")

            dow = pd.to_datetime(day).weekday()  # 0=월
            dow_group = get_dow_group(dow)

            tau_row = hybrid_tau[(hybrid_tau["정류장_순서"]==pivot_seq) &
                                 (hybrid_tau["도착_시간"]==to_hour) &
                                 (hybrid_tau["요일그룹"]==dow_group)]
            if tau_row.empty:
                if debug: print(f"[DEBUG] τ없음 → {day} {to_hour}h seq{pivot_seq} 스킵")
                continue
            tau_min = float(tau_row.iloc[0]["tau_final_min"])
            tau_source = tau_row.iloc[0]["tau_source"]

            moved_total = 0
            for j, drow in don_day.sort_values("spare", ascending=False).iterrows():
                if moved_total>=min(need, max_moves_per_target):
                    break
                if drow["spare"]<=0:
                    continue
                from_hour = int(drow["from_hour"])
                take = int(min(drow["spare"], need - moved_total))
                if take<=0:
                    continue

                pivot_time = pd.to_datetime(str(day)) + pd.Timedelta(hours=to_hour)
                depart_time = pivot_time - pd.Timedelta(minutes=tau_min)
                snap_min = (depart_time.minute // headway_min) * headway_min
                depart_time_snapped = depart_time.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(minutes=snap_min)

                plans.append({
                    "기준_날짜": day,
                    "from_hour": from_hour,
                    "to_hour": to_hour,
                    "pivot_stop_seq": pivot_seq,
                    "pivot_stop_name": pivot_name,
                    "needed_before": int(need),
                    "moved": int(take),
                    "needed_after": int(need - (moved_total + take)),
                    "depart_time_snapped": depart_time_snapped,
                    "tau_min": round(tau_min,3),
                    "tau_source": tau_source
                })

                don_day.loc[j, "spare"] -= take
                moved_total += take

            if debug and moved_total==0:
                print(f"[DEBUG] {day} {to_hour}h seq{pivot_seq}: 도너 부족/제약으로 이동 0대")

    if not plans:
        return pd.DataFrame(columns=[
            "기준_날짜","from_hour","to_hour","pivot_stop_seq","pivot_stop_name",
            "needed_before","moved","needed_after","depart_time_snapped","tau_min","tau_source"
        ])
    return pd.DataFrame(plans).sort_values(["기준_날짜","to_hour","from_hour","pivot_stop_seq"]).reset_index(drop=True)

# ========= 로드 =========
pred = pd.read_csv(PRED_CSV)
hyb  = pd.read_csv(HYBRID_CSV)

# 타입/컬럼 표준화
pred["기준_날짜"] = pd.to_datetime(pred["기준_날짜"]).dt.date
pred["시간"] = pred["시간"].astype(int)
pred["정류장_순서"] = pred["정류장_순서"].astype(int)
if "버스당_onboard_추정" not in pred.columns:
    raise ValueError("PRED_CSV에 '버스당_onboard_추정' 컬럼이 필요합니다.")
if "통과버스수" not in pred.columns:
    raise ValueError("PRED_CSV에 '통과버스수' 컬럼이 필요합니다.")
if "역명" not in pred.columns:
    pred["역명"] = ""

# 하이브리드 τ 인덱스 키 통일
hyb = hyb.rename(columns={"도착_시간":"시간"})
hyb["정류장_순서"] = hyb["정류장_순서"].astype(int)
hyb["시간"] = hyb["시간"].astype(int)

# ---------- 시간대별 혼잡 필요량(need) 산출 ----------
def build_hourly_need_from_pred(pred_df: pd.DataFrame) -> pd.DataFrame:
    g = pred_df.groupby(["기준_날짜","시간","정류장_순서"], as_index=False).agg(
        max_onboard=("버스당_onboard_추정","max"),
        bus_count=("통과버스수","max"),
        stop_name=("역명","max"),
    )
    idx = g.groupby(["기준_날짜","시간"])["max_onboard"].idxmax()
    worst = g.loc[idx].reset_index(drop=True)
    worst["needed_before"] = worst.apply(lambda r: needed_additional_buses(r["max_onboard"], r["bus_count"]), axis=1)
    hourly_need = worst.rename(columns={
        "정류장_순서":"pivot_stop_seq",
        "stop_name":"pivot_stop_name"
    })[["기준_날짜","시간","pivot_stop_seq","pivot_stop_name","needed_before"]]
    return hourly_need

# ---------- 시간대별 공여 가능량(supply) 산출 ----------
def build_hourly_supply_from_pred(pred_df: pd.DataFrame) -> pd.DataFrame:
    g = pred_df.groupby(["기준_날짜","시간","정류장_순서"], as_index=False).agg(
        max_onboard=("버스당_onboard_추정","max"),
        bus_count=("통과버스수","max"),
    )
    # 시간대 전체에서 '최대 온보드'와 '해당 시간 버스수'로 보수적으로 판단
    by_hour = g.groupby(["기준_날짜","시간"], as_index=False).agg(
        max_onboard=("max_onboard","max"),
        bus_count=("bus_count","max")
    )
    need_buses = np.ceil(by_hour["bus_count"] * (by_hour["max_onboard"] / THRESH_OK))
    by_hour["spare"] = (by_hour["bus_count"] - need_buses).clip(lower=0).astype(int)
    return by_hour[["기준_날짜","시간","spare"]]

# ========= 실행: 하이브리드 τ + 보호구간 + 그리디 매칭 기반 재배치 제안 =========
# 1) 시간대별 need/supply 만들기
hourly_need   = build_hourly_need_from_pred(pred)
hourly_supply = build_hourly_supply_from_pred(pred)

# 2) 근사 스케줄(기점의 운영 시간대): 정류장_순서==1 기준
schedule_approx = (
    pred[pred["정류장_순서"]==1][["기준_날짜","시간","정류장_순서"]]
    .drop_duplicates()
    .rename(columns={"시간":"도착_시간"})
)

# 3) 보호구간 적용 후 타깃/도너 추출
tgt, don = build_targets_and_donors(hourly_need, hourly_supply, schedule_approx, protect_h=3, debug=True)

# 4) τ lookup 준비(열명 통일)
hyb_tau = hyb.rename(columns={"시간":"도착_시간"})

# 5) 계획 도출
plans = plan_reallocation(tgt, don, hyb_tau, get_dow_group, headway_min=HEADWAY_GRID_MIN, debug=True)

if plans.empty:
    print("✅ 재배치가 필요한 혼잡 시간대가 발견되지 않았습니다. (또는 도너 없음)")
else:
    plans = plans.sort_values(["기준_날짜","to_hour","from_hour","pivot_stop_seq"]).reset_index(drop=True)
    safe_display(plans, title="\n🧭 재배치 제안(상위 30행):")

    summary = plans.groupby("기준_날짜").agg(
        moved_total=("moved","sum"),
        unique_to_hours=("to_hour","nunique"),
        unique_from_hours=("from_hour","nunique")
    ).reset_index()
    safe_display(summary, title="\n📊 일자별 이동 요약:")