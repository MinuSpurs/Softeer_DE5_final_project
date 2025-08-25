# filename: build_hybrid_tau_and_plan.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

# =========================
# 0) 사용자 설정
# =========================
SCHEDULE_CSV = "/Users/minwoo/Desktop/softeer/data_engineering_course_materials/missions/final_project/result/172_버스단위_스케줄+수요분배_2025-06-01_to_2025-06-30.csv"
# 정류장 i -> i+1 거리(m). 최소 컬럼: [정류장_순서, 링크_구간거리(m)]
LINKS_CSV    = "/Users/minwoo/Desktop/softeer/data_engineering_course_materials/missions/final_project/data/172_링크_구간거리_from_보정.csv"

OUT_TAU_CSV  = "/Users/minwoo/Desktop/softeer/data_engineering_course_materials/missions/final_project/result/hybrid_tau_lookup.csv"

# 관측을 얼마나 모아야 관측기반을 쓰느냐(미만이면 거리합산 fallback)
MIN_OBS = 5
# 관측 분위수(추천: "p60" 또는 "median")
OBS_QUANTILE = "p60"

# 거리합산용 정류장별 정차(dwell) 기본값(분)
DWELL_MIN_PER_STOP = 0.2   # 12초

# 출발 스냅(헤드웨이 격자, 분). 10이면 00,10,20,30,40,50분에 맞춰줌
HEADWAY_GRID_MIN = 10

# =========================
# 1) 속도 프로파일 (시간별 m/분)
# =========================
def speed_m_per_min(hour: int) -> float:
    """
    단순 속도 프로파일:
      - 출퇴근 혼잡(07~09, 17~20): 15 km/h
      - 그 외: 22 km/h
    필요시 여기만 바꾸면 됨.
    """
    if 7 <= hour <= 9 or 17 <= hour <= 20:
        kmh = 15.0
    else:
        kmh = 22.0
    return kmh * 1000.0 / 60.0

# =========================
# 2) 데이터 적재 & 전처리
# =========================
def load_schedule(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 필수 컬럼 체크(최소)
    need_cols = {"bus_id","기준_날짜","정류장_순서","도착시각"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"SCHEDULE_CSV에 필요한 컬럼이 없습니다: {missing}")

    # 타입 정리
    df["도착시각"] = pd.to_datetime(df["도착시각"])
    # '기준_날짜'를 날짜로 통일
    df["기준_날짜"] = pd.to_datetime(df["기준_날짜"]).dt.date
    # 도착_시간(0~23) 재계산(있으면 덮어씀)
    df["도착_시간"] = df["도착시각"].dt.hour
    # 평일/주말 구분
    dow = pd.to_datetime(df["기준_날짜"]).dt.weekday  # 월=0 … 일=6
    df["요일그룹"] = np.where(dow < 5, "평일", "주말")
    # 정렬 보장
    df = df.sort_values(["bus_id","정류장_순서","도착시각"]).reset_index(drop=True)
    return df

def load_links(path: str) -> pd.DataFrame:
    links = pd.read_csv(path)
    # 허용 스키마: (1) [정류장_순서, 링크_구간거리(m)]  (정류장_순서=i의 "i→i+1" 거리)
    need_cols = {"정류장_순서","링크_구간거리(m)"}
    missing = need_cols - set(links.columns)
    if missing:
        raise ValueError(f"LINKS_CSV에 필요한 컬럼이 없습니다: {missing}")
    # 정렬
    links = links.sort_values("정류장_순서").reset_index(drop=True)
    return links

# =========================
# 3) 관측기반 τ(1→k) 계산
# =========================
def build_observed_tau(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    관측 기반으로 정류장1→k 소요시간(분)을 시간·요일그룹별로 집계
    - 방향 통일: bus_id별 첫 정류장_순서 == 1 인 운행만 사용
    - 그룹: [정류장_순서, 도착_시간, 요일그룹]
    - 이상치 완화: 그룹 내 5~95 분위로 clip, 그 후 median/p60/p75 구함
    """
    # bus_id별 '출발'이 1번인 운행만 추림
    first_seq = (
        schedule.sort_values(["bus_id","정류장_순서"])
        .groupby("bus_id", as_index=False)
        .first()[["bus_id","정류장_순서"]]
        .rename(columns={"정류장_순서":"trip_first_seq"})
    )
    df = schedule.merge(first_seq, on="bus_id", how="left")
    df = df[df["trip_first_seq"] == 1].copy()

    # 각 운행의 기준시각 t0(정류장1 도착시각)
    t0 = (
        df[df["정류장_순서"]==1][["bus_id","도착시각"]]
        .rename(columns={"도착시각":"t0"})
    )
    df = df.merge(t0, on="bus_id", how="left")

    # 소요시간(분) = (정류장k 도착시각 - t0)
    df["elapsed_min"] = (df["도착시각"] - df["t0"]).dt.total_seconds()/60.0
    # 자기 자신(정류장1)은 0.0
    df.loc[df["정류장_순서"]==1, "elapsed_min"] = 0.0

    # 그룹 집계
    def agg_group(g: pd.DataFrame) -> pd.Series:
        x = g["elapsed_min"].dropna().values
        n = len(x)
        if n == 0:
            return pd.Series({"obs_n":0,"tau_med":np.nan,"tau_p60":np.nan,"tau_p75":np.nan})
        # 5~95 분위로 clip
        lo, hi = np.quantile(x, [0.05, 0.95]) if n >= 20 else (np.min(x), np.max(x))
        x = np.clip(x, lo, hi)
        return pd.Series({
            "obs_n": n,
            "tau_med": float(np.median(x)),
            "tau_p60": float(np.quantile(x, 0.60)),
            "tau_p75": float(np.quantile(x, 0.75))
        })

    obs = (
        df.groupby(["정류장_순서","도착_시간","요일그룹"], as_index=False)
          .apply(agg_group)
          .reset_index(drop=True)
    )
    return obs

# =========================
# 4) 거리합산 fallback τ 계산
# =========================
def build_fallback_tau(links: pd.DataFrame) -> pd.DataFrame:
    """
    정류장1→k 거리 누적 후 시간별 속도로 나눠 τ 계산 + 정차(dwell) 보정
    - k=1이면 τ=0
    - dwell은 (k-1) * DWELL_MIN_PER_STOP
    - 요일그룹은 평일/주말 모두 동일(관측 없을 때만 쓰는 백업이라 단순화)
    """
    # 누적거리: 정류장_순서 1→2, 2→3, ... (i행이 i→i+1 거리)
    links = links.sort_values("정류장_순서").reset_index(drop=True)
    # k별 누적거리: sum_{i=1}^{k-1} link_dist[i]
    links["cum_dist_m"] = links["링크_구간거리(m)"].cumsum().shift(1, fill_value=0.0)
    # k값은 links의 정류장_순서 그대로 사용 (k=1이면 cum_dist=0)
    base = links[["정류장_순서","cum_dist_m"]].copy()

    rows = []
    for hour in range(24):
        v = max(speed_m_per_min(hour), 1e-6)
        tmp = base.copy()
        tmp["시간"] = hour
        # dwell: (k-1)*DWELL
        tmp["tau_fallback_min"] = (tmp["cum_dist_m"]/v) + (tmp["정류장_순서"]-1).clip(lower=0)*DWELL_MIN_PER_STOP
        tmp["요일그룹"] = "평일"  # 관측 없을 때만 쓰는 백업이라 단순화
        rows.append(tmp)
        tmp2 = tmp.copy()
        tmp2["요일그룹"] = "주말"
        rows.append(tmp2)
    fb = pd.concat(rows, ignore_index=True)
    fb = fb.rename(columns={"시간":"도착_시간"})
    return fb[["정류장_순서","도착_시간","요일그룹","tau_fallback_min"]]

# =========================
# 5) 하이브리드 결합
# =========================
def combine_hybrid(obs: pd.DataFrame, fb: pd.DataFrame) -> pd.DataFrame:
    df = fb.merge(obs, on=["정류장_순서","도착_시간","요일그룹"], how="left")

    # 어떤 분위수를 쓸지 선택
    if OBS_QUANTILE == "p60":
        chosen = "tau_p60"
    elif OBS_QUANTILE == "p75":
        chosen = "tau_p75"
    else:
        chosen = "tau_med"

    # 최종 τ 및 출처
    df["tau_final_min"] = np.where(
        (df["obs_n"].fillna(0) >= MIN_OBS) & df[chosen].notna(),
        df[chosen],
        df["tau_fallback_min"]
    )
    df["tau_source"] = np.where(
        (df["obs_n"].fillna(0) >= MIN_OBS) & df[chosen].notna(),
        f"observed_{OBS_QUANTILE}",
        "fallback"
    )

    # 보기 좋게 정리
    out = df[[
        "정류장_순서","도착_시간","요일그룹",
        "tau_final_min","tau_source",
        "obs_n","tau_med","tau_p60","tau_p75","tau_fallback_min"
    ]].sort_values(["정류장_순서","요일그룹","도착_시간"])
    return out

# =========================
# 6) 도착 목표 → 출발시각 산출(예시)
# =========================
@dataclass
class PlanResult:
    target_date: str
    target_hour: int
    stop_seq: int
    arrival_minute_in_hour: int
    tau_used_min: float
    tau_source: str
    depart_time_str: str
    depart_time_snapped_str: str

def snap_to_grid_minute(minute: int, grid: int) -> int:
    """분 단위를 가장 가까운 그리드로 스냅(아래쪽으로 내림)"""
    if grid <= 1:
        return minute
    return (minute // grid) * grid

def plan_departure_for_arrival(hybrid_tau: pd.DataFrame,
                               date_str: str,
                               target_hour: int,
                               stop_seq: int,
                               arrival_minute_in_hour: int = 30) -> PlanResult:
    """
    주어진 날짜/시단/정류장에 'HH:MM 도착'을 맞추려면
    '정류장1 출발'은 몇 시 몇 분이어야 하는지 계산(스냅 포함)
    """
    day = pd.to_datetime(date_str).date()
    dow = pd.to_datetime(day).weekday()
    yoil = "평일" if dow < 5 else "주말"

    key = (stop_seq, target_hour, yoil)
    row = hybrid_tau[
        (hybrid_tau["정류장_순서"]==stop_seq) &
        (hybrid_tau["도착_시간"]==target_hour) &
        (hybrid_tau["요일그룹"]==yoil)
    ]
    if row.empty:
        raise ValueError(f"hybrid_tau에서 키 {key} 를 찾지 못했습니다.")
    tau = float(row["tau_final_min"].iloc[0])
    src = str(row["tau_source"].iloc[0])

    # 목표 도착시각
    arrival = pd.Timestamp(f"{date_str} {target_hour:02d}:{arrival_minute_in_hour:02d}:00")
    depart = arrival - pd.Timedelta(minutes=tau)

    # 헤드웨이 격자 스냅(분만 조정)
    snapped_minute = snap_to_grid_minute(depart.minute, HEADWAY_GRID_MIN)
    depart_snapped = depart.replace(minute=snapped_minute, second=0, microsecond=0)

    return PlanResult(
        target_date=date_str,
        target_hour=target_hour,
        stop_seq=stop_seq,
        arrival_minute_in_hour=arrival_minute_in_hour,
        tau_used_min=round(tau, 3),
        tau_source=src,
        depart_time_str=str(depart),
        depart_time_snapped_str=str(depart_snapped),
    )

# =========================
# 7) 메인
# =========================
def main():
    print("📥 Load schedule…")
    sched = load_schedule(SCHEDULE_CSV)

    print("📥 Load link distances…")
    links = load_links(LINKS_CSV)

    print("🧮 Build observed τ(1→k)…")
    obs_tau = build_observed_tau(sched)

    print("🧮 Build fallback τ via distance+speed…")
    fb_tau = build_fallback_tau(links)

    print("🔗 Combine hybrid…")
    hybrid = combine_hybrid(obs_tau, fb_tau)

    os.makedirs(os.path.dirname(OUT_TAU_CSV), exist_ok=True)
    hybrid.to_csv(OUT_TAU_CSV, index=False)
    print(f"✅ Saved hybrid τ lookup: {OUT_TAU_CSV}")
    print(hybrid.head(12))

    # ===== 예시: 2025-06-26 18시, 정류장 79(창경궁) 18:30 도착 목표 =====
    try:
        example = plan_departure_for_arrival(
            hybrid_tau=hybrid,
            date_str="2025-06-26",
            target_hour=18,
            stop_seq=79,                 # 창경궁(00079)
            arrival_minute_in_hour=30    # 18:30 도착 목표
        )
        print("\n[예시] 2025-06-26, 18시, 정류장 79 도착 18:30 맞추기")
        print(f"- 사용 τ(min): {example.tau_used_min}  (source={example.tau_source})")
        print(f"- 계산된 출발시각: {example.depart_time_str}")
        print(f"- 헤드웨이 {HEADWAY_GRID_MIN}분 스냅: {example.depart_time_snapped_str}")
    except Exception as e:
        print(f"[예시 스킵] {e}")

if __name__ == "__main__":
    main()