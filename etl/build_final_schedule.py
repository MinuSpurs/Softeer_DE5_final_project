import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import argparse

# ========= 설정 =========
PRED_CSV = "/Users/minwoo/Desktop/softeer/data_engineering_course_materials/missions/final_project/data/172_LSTM_preds_정류장별_시간별_승하차_예측_혼잡도+추가배차.csv"
LINK_CSV = "/Users/minwoo/Desktop/softeer/data_engineering_course_materials/missions/final_project/data/172_시간대별_정류장_노선_추정승하차+운행횟수+순서+링크거리.csv"  # 링크거리 CSV가 있으면 경로 문자열로 바꿔주세요. 없으면 None 그대로 두면 됩니다.
DEFAULT_LINK_M = 300.0  # 링크거리 기본값(정류장_순서 1은 자동으로 0m)

OUT_DIR = "/Users/minwoo/Desktop/softeer/data_engineering_course_materials/missions/final_project/result"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

START_DATE = "2025-06-01"
END_DATE   = "2025-06-30"

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS fact_final_schedule (
  bus_id TEXT,
  date TEXT,
  stop_seq INTEGER,
  stop_id TEXT,
  stop_name TEXT,
  arrival_time TEXT,
  arrival_hour INTEGER
);

CREATE TABLE IF NOT EXISTS fact_final_allocation (
  bus_id TEXT,
  date TEXT,
  stop_seq INTEGER,
  stop_id TEXT,
  stop_name TEXT,
  arrival_time TEXT,
  arrival_hour INTEGER,
  board_total REAL,
  alight_total REAL,
  board_per_bus REAL,
  alight_per_bus REAL,
  onboard_before REAL,
  onboard_after REAL,
  crowding TEXT
);
"""

# ========= 속도 프로파일 (m/min) =========
def speed_m_per_min(h):
    # 출퇴근 시간 느림, 그 외 빠름 (필요 시 조정)
    kmh = 15 if (7 <= h <= 9 or 17 <= h <= 19) else 22
    return kmh * 1000.0 / 60.0

# ========= 데이터 로드 =========
pred = pd.read_csv(PRED_CSV)

# 필수 컬럼 확인
need_cols = ["기준_날짜","시간","정류장_ID","정류장_순서","역명","통과버스수"]
missing = [c for c in need_cols if c not in pred.columns]
if missing:
    raise ValueError(f"예측 CSV에 필요한 컬럼이 없습니다: {missing}")

# 날짜 범위 필터
pred["기준_날짜"] = pd.to_datetime(pred["기준_날짜"]).dt.strftime("%Y-%m-%d")
if START_DATE and END_DATE:
    mask = (pred["기준_날짜"] >= START_DATE) & (pred["기준_날짜"] <= END_DATE)
    pred = pred.loc[mask].copy()
days = sorted(pred["기준_날짜"].unique())
if not days:
    raise ValueError("처리할 날짜가 없습니다. START_DATE/END_DATE를 확인하세요.")

# ========= 유틸: 예측 컬럼 이름 해석 =========
def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ========= 링크 테이블 준비 (옵션: 파일 / 기본값 생성) =========
def build_link_table(day_df: pd.DataFrame) -> tuple[pd.DataFrame, float, str]:
    """
    반환: (정류장_순서, 링크거리m), 평균(1번 제외), 소스표시
    - LINK_CSV가 있으면 파일 사용(정류장_순서별 평균)
    - 없으면 day_df의 정류장_순서로 기본값 생성(1번은 0, 그 외 DEFAULT_LINK_M)
    - 1번 정류장_순서는 0m, 그 외는 결측 또는 0인 경우 평균값으로 대체됨
    """
    src = "default(300m)"
    if LINK_CSV:
        link_raw = pd.read_csv(LINK_CSV)
        # 링크 컬럼 찾기
        link_dist_col = None
        for c in link_raw.columns:
            if "링크" in c and "거리" in c:
                link_dist_col = c
                break
        if link_dist_col is None:
            raise ValueError("LINK_CSV에서 링크거리 컬럼(예: '링크_구간거리(m)')을 찾지 못했습니다.")
        link_small = link_raw.groupby("정류장_순서", as_index=False)[link_dist_col].mean()
        # 1번=0, 그 외 결측/0은 평균 대체
        if (link_small["정류장_순서"] == 1).any():
            link_small.loc[link_small["정류장_순서"] == 1, link_dist_col] = 0.0
        mean_nonzero = link_small.loc[link_small["정류장_순서"] != 1, link_dist_col].replace(0, np.nan).mean()
        link_small[link_dist_col] = link_small[link_dist_col].fillna(mean_nonzero)
        mask_zero = (link_small["정류장_순서"] != 1) & (link_small[link_dist_col] <= 0)
        link_small.loc[mask_zero, link_dist_col] = mean_nonzero
        # 해당 일자 순서만
        wanted = sorted(day_df["정류장_순서"].unique().tolist())
        link_small = link_small[link_small["정류장_순서"].isin(wanted)].copy()
        src = "LINK_CSV"
        return link_small.rename(columns={link_dist_col:"링크_구간거리(m)"}), float(mean_nonzero), src
    else:
        wanted = sorted(day_df["정류장_순서"].unique().tolist())
        rows = []
        for s in wanted:
            rows.append({"정류장_순서": s, "링크_구간거리(m)": (0.0 if s == 1 else DEFAULT_LINK_M)})
        link_small = pd.DataFrame(rows)
        mean_nonzero = DEFAULT_LINK_M
        return link_small, mean_nonzero, src

# ========= 출발 스케줄 전개 =========
def expand_departures(date_str: str, first_stop_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    정류장_순서==1 에서 시간별 통과버스수를 헤드웨이로 나눠 버스 단위 출발시각 생성
    """
    out = []
    for _, r in first_stop_hourly.iterrows():
        hh = int(r["시간"])
        n = int(round(r["통과버스수"] if pd.notna(r["통과버스수"]) else 0))
        if n <= 0:
            continue
        headway = 60.0 / n
        dt0 = datetime.strptime(f"{date_str} {hh:02d}:00:00", "%Y-%m-%d %H:%M:%S")
        for k in range(n):
            dep_time = dt0 + timedelta(minutes=headway * k)
            out.append({
                "bus_id": f"{date_str.replace('-','')}-{hh:02d}-{k+1:02d}",
                "기준_날짜": date_str,
                "출발시각": dep_time,
                "출발_시간": hh,
                "출발_헤드웨이_분": headway
            })
    return pd.DataFrame(out)

# ========= 링크거리 → 각 정류장 도착시각 전파 =========
def propagate_schedule(departures: pd.DataFrame,
                       stop_orders: list[int],
                       link_table: pd.DataFrame,
                       mean_nonzero: float) -> pd.DataFrame:
    link_map = dict(zip(link_table["정류장_순서"], link_table["링크_구간거리(m)"]))
    rows = []
    for _, d in departures.iterrows():
        t0 = d["출발시각"]
        arr = t0
        for s in stop_orders:
            if s == 1:
                arr = t0
            else:
                dist_m = float(link_map.get(s, mean_nonzero))
                v = speed_m_per_min(arr.hour)
                travel_min = dist_m / max(v, 1e-6)
                arr = arr + timedelta(minutes=travel_min)
            rows.append({
                "bus_id": d["bus_id"],
                "기준_날짜": d["기준_날짜"],
                "정류장_순서": s,
                "도착시각": arr,
                "도착_시간": arr.hour
            })
    return pd.DataFrame(rows)

# ========= 하루 처리 =========
def process_one_day(date_str: str, conn, table_schedule, table_alloc):
    day = pred[pred["기준_날짜"] == date_str].copy()
    if day.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 링크 테이블
    link_small, mean_nonzero, link_src = build_link_table(day)
    stop_orders = sorted(day["정류장_순서"].unique().tolist())

    # 출발(순서 1)
    first_stop = day[day["정류장_순서"] == 1][["기준_날짜","시간","통과버스수"]].copy()
    deps = expand_departures(date_str, first_stop)
    if deps.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 스케줄 전파
    schedule = propagate_schedule(deps, stop_orders, link_small, mean_nonzero)

    # 정류장 메타 붙이기
    ref = day.sort_values("정류장_순서")[["정류장_순서","정류장_ID","역명"]].drop_duplicates("정류장_순서")
    schedule = schedule.merge(ref, on="정류장_순서", how="left")

    # 시단 버킷 및 버스수
    schedule["도착_시단"] = schedule["도착시각"].dt.floor("h")  # FutureWarning 방지용 소문자 h
    bus_cnt = schedule.groupby(["기준_날짜","도착_시단","정류장_순서"], as_index=False)["bus_id"].count() \
                      .rename(columns={"bus_id":"그_시간_버스대수"})

    # 시간·순서별 총 승차/하차 (예측 파일에서 직접 사용; 없으면 0)
    day_tmp = day[["기준_날짜","시간","정류장_순서","정류장_ID","역명","통과버스수"]].copy()
    # 통과버스수 결측 보호
    day_tmp["통과버스수"] = pd.to_numeric(day_tmp["통과버스수"], errors="coerce").fillna(0.0)

    # 총량/버스당 후보 컬럼명 (현재 CSV 스펙 반영)
    board_total_col  = first_existing(day, ["예측_승차_총","예측_승차인원","승차인원","승차","총승차"])
    alight_total_col = first_existing(day, ["예측_하차_총","예측_하차인원","하차인원","하차","총하차"])
    board_perbus_col = first_existing(day, ["버스당_승차","예측_버스당_승차","버스당_탑승예측","개선후_버스당"])
    alight_perbus_col= first_existing(day, ["버스당_하차","예측_버스당_하차"])

    # 총 승차 계산: 1) 총량형이 있으면 그걸 사용, 2) 없고 버스당형만 있으면 통과버스수로 환산
    if board_total_col is not None:
        day_tmp["총승차"] = pd.to_numeric(day[board_total_col], errors="coerce").fillna(0.0).clip(lower=0)
    elif board_perbus_col is not None:
        perb = pd.to_numeric(day[board_perbus_col], errors="coerce").fillna(0.0)
        day_tmp["총승차"] = (perb * day_tmp["통과버스수"]).clip(lower=0)
    else:
        day_tmp["총승차"] = 0.0

    # 총 하차 계산
    if alight_total_col is not None:
        day_tmp["총하차"] = pd.to_numeric(day[alight_total_col], errors="coerce").fillna(0.0).clip(lower=0)
    elif alight_perbus_col is not None:
        perb = pd.to_numeric(day[alight_perbus_col], errors="coerce").fillna(0.0)
        day_tmp["총하차"] = (perb * day_tmp["통과버스수"]).clip(lower=0)
    else:
        day_tmp["총하차"] = 0.0

    # 시단 키 생성 (병합 키용)
    day_tmp["도착_시단"] = pd.to_datetime(day_tmp["기준_날짜"] + " " + day_tmp["시간"].astype(int).astype(str) + ":00:00")

    # 분배 및 탑승객 추적
    alloc = schedule.merge(bus_cnt, on=["기준_날짜","도착_시단","정류장_순서"], how="left")
    alloc = alloc.merge(day_tmp[["기준_날짜","도착_시단","정류장_순서","정류장_ID","역명","통과버스수","총승차","총하차"]],
                        on=["기준_날짜","도착_시단","정류장_순서"], how="left")

    # 버스당 승하차 균등 분배 (0 나눗셈 보호)
    denom = alloc["그_시간_버스대수"].replace(0, np.nan)
    alloc["버스당_할당승차"] = (alloc["총승차"] / denom).fillna(0.0).clip(lower=0)
    alloc["버스당_할당하차"] = (alloc["총하차"] / denom).fillna(0.0).clip(lower=0)

    # 버스별 누적 탑승 인원 추적: onboard_after = cumsum(승차-하차), 음수 방지
    alloc = alloc.sort_values(["bus_id","정류장_순서"]).reset_index(drop=True)
    alloc["delta"] = alloc["버스당_할당승차"] - alloc["버스당_할당하차"]
    alloc["onboard_after"] = alloc.groupby("bus_id")["delta"].cumsum().clip(lower=0)
    alloc["onboard_before"] = (alloc["onboard_after"] - alloc["delta"]).clip(lower=0)

    # 좌석 33 기준 혼잡도: ≤33 여유, ≤49.5 보통, >49.5 혼잡
    CAP = 33.0
    alloc["혼잡도"] = np.where(
        alloc["onboard_after"] <= CAP,
        "여유",
        np.where(alloc["onboard_after"] <= CAP*1.5, "보통", "혼잡")
    )

    # 편의 칼럼
    alloc["총수요(근사)"] = alloc["총승차"]  # 보고용 (필요시 총승차+총하차로 변경)
    alloc["링크거리_소스"] = ("LINK_CSV" if LINK_CSV else f"default({DEFAULT_LINK_M:.0f}m)")

    # 저장
    sch_path = f"{OUT_DIR}/172_버스단위_스케줄_{date_str}.csv"
    alloc_path = f"{OUT_DIR}/172_버스단위_스케줄+수요분배_{date_str}.csv"
    schedule.to_csv(sch_path, index=False, encoding="utf-8-sig")
    alloc.to_csv(alloc_path, index=False, encoding="utf-8-sig")

    # DB 저장
    cur = conn.cursor()
    # Insert schedule data
    schedule_records = []
    for _, row in schedule.iterrows():
        schedule_records.append((
            row["bus_id"],
            row["기준_날짜"],
            int(row["정류장_순서"]),
            row["정류장_ID"],
            row["역명"],
            row["도착시각"].strftime("%Y-%m-%d %H:%M:%S"),
            int(row["도착_시간"])
        ))
    cur.executemany(f"""
        INSERT INTO {table_schedule} (bus_id, date, stop_seq, stop_id, stop_name, arrival_time, arrival_hour)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, schedule_records)

    # Insert allocation data
    alloc_records = []
    for _, row in alloc.iterrows():
        alloc_records.append((
            row["bus_id"],
            row["기준_날짜"],
            int(row["정류장_순서"]),
            row["정류장_ID"],
            row["역명"],
            row["도착시각"].strftime("%Y-%m-%d %H:%M:%S"),
            int(row["도착_시간"]),
            float(row["총승차"]),
            float(row["총하차"]),
            float(row["버스당_할당승차"]),
            float(row["버스당_할당하차"]),
            float(row["onboard_before"]),
            float(row["onboard_after"]),
            row["혼잡도"]
        ))
    cur.executemany(f"""
        INSERT INTO {table_alloc} (bus_id, date, stop_seq, stop_id, stop_name, arrival_time, arrival_hour,
            board_total, alight_total, board_per_bus, alight_per_bus, onboard_before, onboard_after, crowding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, alloc_records)

    conn.commit()
    return schedule, alloc

def main():
    parser = argparse.ArgumentParser(description="Build final bus schedule and save to CSV and SQLite DB.")
    parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite database file.")
    parser.add_argument("--table-schedule", type=str, default="fact_final_schedule", help="Table name for schedule data.")
    parser.add_argument("--table-alloc", type=str, default="fact_final_allocation", help="Table name for allocation data.")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    cur = conn.cursor()
    cur.executescript(CREATE_TABLES_SQL)
    conn.commit()

    all_sch, all_alloc = [], []
    for d in days:
        sch, alc = process_one_day(d, conn, args.table_schedule, args.table_alloc)
        if not sch.empty:
            all_sch.append(sch)
        if not alc.empty:
            all_alloc.append(alc)

    if all_sch:
        msch = pd.concat(all_sch, ignore_index=True)
        msch_path = f"{OUT_DIR}/172_버스단위_스케줄_{START_DATE}_to_{END_DATE}.csv"
        msch.to_csv(msch_path, index=False, encoding="utf-8-sig")
        print("✅ 월간 스케줄 저장:", msch_path)

    if all_alloc:
        malc = pd.concat(all_alloc, ignore_index=True)
        malc_path = f"{OUT_DIR}/172_버스단위_스케줄+수요분배_{START_DATE}_to_{END_DATE}.csv"
        malc.to_csv(malc_path, index=False, encoding="utf-8-sig")
        print("✅ 월간 스케줄+수요분배 저장:", malc_path)

    conn.close()
    print(f"처리 일수: {len(days)}일 (범위: {START_DATE} ~ {END_DATE})")

if __name__ == "__main__":
    main()