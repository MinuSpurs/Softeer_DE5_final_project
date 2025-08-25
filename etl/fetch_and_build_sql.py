# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pandas as pd

# ----------------------------
# 유틸
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_xml(text: str, path: Path):
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")

def get_text(elem, tag, default=""):
    v = elem.findtext(tag)
    return v if v is not None else default

def to_int_safe(v, default=0):
    try:
        return int(str(v).strip())
    except Exception:
        return default

# ----------------------------
# API 호출 (requests 없이 urllib 사용: 표준 라이브러리)
# ----------------------------
from urllib.request import urlopen
from urllib.parse import quote

def fetch_url(url: str) -> str:
    with urlopen(url) as resp:
        return resp.read().decode("utf-8")

def _q(x):
    # quote path segment safely (handles non-ASCII like API keys or route numbers)
    return quote(str(x), safe="")

# ----------------------------
# 노선 매핑 적재 함수 추가
# ----------------------------
def upsert_dim_route_map(conn, map_csv_path: Path):
    """
    route_key(노선_ID) ↔ rte_id/RTE_NO 매핑 적재.
    CSV 컬럼 요구: ['노선_ID','RTE_ID','RTE_NO']
    """
    if not map_csv_path.exists():
        print(f"[warn] route map CSV not found: {map_csv_path} (skip)")
        return
    df = pd.read_csv(map_csv_path)
    need = ["노선_ID", "RTE_ID", "RTE_NO"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"route_key_map CSV에 '{c}' 컬럼이 필요합니다.")
    df = df.rename(columns={
        "노선_ID": "route_key",
        "RTE_ID":  "rte_id",
        "RTE_NO":  "route_no"
    })
    # 문자열 정규화
    df["route_key"] = df["route_key"].astype(str).str.strip()
    df["rte_id"]    = df["rte_id"].astype(str).str.strip()
    df["route_no"]  = df["route_no"].astype(str).str.strip()

    tuples = list(df[["route_key","rte_id","route_no"]].itertuples(index=False, name=None))
    conn.executemany("""
        INSERT OR REPLACE INTO dim_route_map(route_key, rte_id, route_no)
        VALUES (?, ?, ?)
    """, tuples)

# ----------------------------
# 파서: CardBusStatisticsServiceNew (일별)
# URL: /xml/CardBusStatisticsServiceNew/1/500/{YYYYMMDD}/{RTE_NO}/
# ----------------------------
def fetch_parse_daily_stats(api_key: str, date_yyyymmdd: str, route_no: str, raw_dir: Path):
    base = f"http://openapi.seoul.go.kr:8088/{_q(api_key)}/xml/CardBusStatisticsServiceNew"
    page_start, page_end = 1, 500
    rows = []
    # 1페이지 호출
    url = f"{base}/{_q(page_start)}/{_q(page_end)}/{_q(date_yyyymmdd)}/{_q(route_no)}/"
    xml = fetch_url(url)
    save_xml(xml, raw_dir / f"{date_yyyymmdd}.xml")

    root = ET.fromstring(xml)
    total = int(root.findtext("list_total_count", default="0"))

    def parse_rows(root_elem):
        for row in root_elem.findall("row"):
            yield {
                "date": get_text(row, "USE_YMD"),        # YYYYMMDD
                "rte_id": get_text(row, "RTE_ID"),
                "route_no": get_text(row, "RTE_NO"),
                "stop_id": get_text(row, "STOPS_ID"),
                "stop_name": get_text(row, "SBWY_STNS_NM"),
                "board_total": to_int_safe(get_text(row, "GTON_TNOPE")),
                "alight_total": to_int_safe(get_text(row, "GTOFF_TNOPE")),
                "reg_ymd": get_text(row, "REG_YMD"),
            }

    rows.extend(list(parse_rows(root)))

    # 페이징 필요 시(여기선 total<=500일 가능성 높지만 방어)
    while len(rows) < total:
        page_start = page_end + 1
        page_end = page_start + 499
        url = f"{base}/{_q(page_start)}/{_q(page_end)}/{_q(date_yyyymmdd)}/{_q(route_no)}/"
        xml = fetch_url(url)
        save_xml(xml, raw_dir / f"{date_yyyymmdd}_{page_start}_{page_end}.xml")
        root = ET.fromstring(xml)
        rows.extend(list(parse_rows(root)))

    return rows

# ----------------------------
# 파서: CardBusTimeNew (월별 시간대)
# URL: /xml/CardBusTimeNew/1/500/{YYYYMM}/{RTE_NO}/
# ----------------------------
def fetch_parse_monthly_time(api_key: str, yyyymm: str, route_no: str, raw_dir: Path):
    base = f"http://openapi.seoul.go.kr:8088/{_q(api_key)}/xml/CardBusTimeNew"
    page_start, page_end = 1, 500
    rows = []

    url = f"{base}/{_q(page_start)}/{_q(page_end)}/{_q(yyyymm)}/{_q(route_no)}/"
    xml = fetch_url(url)
    save_xml(xml, raw_dir / f"{yyyymm}.xml")

    root = ET.fromstring(xml)
    total = int(root.findtext("list_total_count", default="0"))

    def get_hour_value(row_elem, h, onoff="ON"):
        # 일부 필드는 TNOPE, 일부는 NOPE로 찍히는 사례가 있어 둘 다 방어
        if onoff == "ON":
            tags = [f"HR_{h}_GET_ON_TNOPE", f"HR_{h}_GET_ON_NOPE"]
        else:
            tags = [f"HR_{h}_GET_OFF_TNOPE", f"HR_{h}_GET_OFF_NOPE"]
        for t in tags:
            v = row_elem.findtext(t)
            if v is not None:
                return to_int_safe(v)
        return 0

    def parse_rows(root_elem):
        for row in root_elem.findall("row"):
            base_rec = {
                "use_ym": get_text(row, "USE_YM"),          # YYYYMM
                "route_no": get_text(row, "RTE_NO"),
                "stop_id": get_text(row, "STOPS_ID"),
                "stop_name": get_text(row, "SBWY_STNS_NM"),
                "reg_ymd": get_text(row, "REG_YMD"),
            }
            # 0~23시 long 정규화
            for h in range(24):
                yield {
                    **base_rec,
                    "hour": h,
                    "board_total": get_hour_value(row, h, "ON"),
                    "alight_total": get_hour_value(row, h, "OFF"),
                }

    rows.extend(list(parse_rows(root)))

    # 페이징 필요시
    while len(rows) < total * 24:  # 한 row당 24시간 레코드가 생성됨
        page_start = page_end + 1
        page_end = page_start + 499
        url = f"{base}/{_q(page_start)}/{_q(page_end)}/{_q(yyyymm)}/{_q(route_no)}/"
        xml = fetch_url(url)
        save_xml(xml, raw_dir / f"{yyyymm}_{page_start}_{page_end}.xml")
        root = ET.fromstring(xml)
        rows.extend(list(parse_rows(root)))

    return rows

# ----------------------------
# DB 초기화 및 적재
# ----------------------------
CREATE_TABLES_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS dim_route_map (
  route_key TEXT PRIMARY KEY,  -- 링크 CSV의 노선_ID
  rte_id    TEXT NOT NULL,     -- 서울 API의 RTE_ID
  route_no  TEXT NOT NULL      -- 서울 API의 RTE_NO (예: '172')
);
CREATE INDEX IF NOT EXISTS idx_route_map_rte ON dim_route_map(rte_id);
CREATE INDEX IF NOT EXISTS idx_route_map_no  ON dim_route_map(route_no);

CREATE TABLE IF NOT EXISTS dim_stop_link (
  route_key TEXT,
  stop_id   TEXT,
  link_distance_m REAL,
  stop_seq  INTEGER,
  PRIMARY KEY(route_key, stop_id, stop_seq)
);

CREATE TABLE IF NOT EXISTS fact_demand_daily_stop (
  date TEXT,            -- YYYYMMDD
  rte_id TEXT,
  route_no TEXT,
  stop_id TEXT,
  stop_name TEXT,
  board_total INTEGER,
  alight_total INTEGER,
  reg_ymd TEXT
);

CREATE INDEX IF NOT EXISTS idx_daily_route_date ON fact_demand_daily_stop(route_no, date);
CREATE INDEX IF NOT EXISTS idx_daily_stop ON fact_demand_daily_stop(stop_id);

CREATE TABLE IF NOT EXISTS fact_demand_monthly_hourly_stop (
  use_ym TEXT,          -- YYYYMM
  route_no TEXT,
  stop_id TEXT,
  stop_name TEXT,
  hour INTEGER,         -- 0~23
  board_total INTEGER,
  alight_total INTEGER,
  reg_ymd TEXT
);

CREATE INDEX IF NOT EXISTS idx_monthly_route_hour ON fact_demand_monthly_hourly_stop(route_no, use_ym, hour);
CREATE INDEX IF NOT EXISTS idx_monthly_stop ON fact_demand_monthly_hourly_stop(stop_id);
"""

def upsert_dim_stop_link(conn, link_csv_path: Path):
    df = pd.read_csv(link_csv_path)
    # 컬럼 명 매핑 방어
    col_map = {
        "노선_ID": "route_key",
        "정류장_ID": "stop_id",
        "링크_구간거리(m)": "link_distance_m",
        "정류장_순서": "stop_seq",
    }
    for k, v in col_map.items():
        if k not in df.columns:
            raise ValueError(f"링크 CSV에 '{k}' 컬럼이 필요합니다.")
    df = df.rename(columns=col_map)
    df["route_key"] = df["route_key"].astype(str)
    df["stop_id"] = df["stop_id"].astype(str)

    tuples = list(df[["route_key","stop_id","link_distance_m","stop_seq"]].itertuples(index=False, name=None))
    conn.executemany("""
        INSERT OR REPLACE INTO dim_stop_link (route_key, stop_id, link_distance_m, stop_seq)
        VALUES (?, ?, ?, ?)
    """, tuples)

def insert_daily(conn, rows):
    tuples = [(
        r["date"], r["rte_id"], r["route_no"], r["stop_id"], r["stop_name"],
        r["board_total"], r["alight_total"], r["reg_ymd"]
    ) for r in rows]
    conn.executemany("""
        INSERT INTO fact_demand_daily_stop
        (date, rte_id, route_no, stop_id, stop_name, board_total, alight_total, reg_ymd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, tuples)

def insert_monthly_hourly(conn, rows):
    tuples = [(
        r["use_ym"], r["route_no"], r["stop_id"], r["stop_name"], r["hour"],
        r["board_total"], r["alight_total"], r["reg_ymd"]
    ) for r in rows]
    conn.executemany("""
        INSERT INTO fact_demand_monthly_hourly_stop
        (use_ym, route_no, stop_id, stop_name, hour, board_total, alight_total, reg_ymd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, tuples)

# ----------------------------
# 메인
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", required=True, help="서울열린데이터 API Key")
    ap.add_argument("--route-no", required=True, help="노선번호 (예: 172)")
    ap.add_argument("--month", required=True, help="YYYYMM (예: 202506)")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD (예: 2025-06-01)")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD (예: 2025-06-30)")
    ap.add_argument("--link-csv", default="data/external/link_distance.csv")
    ap.add_argument("--route-map-csv", default="data/external/route_key_map.csv",
                    help="노선_ID(링크) ↔ RTE_ID/RTE_NO(서울API) 매핑 CSV")
    ap.add_argument("--db-path", default="db/seoul_bus_172.db")
    ap.add_argument("--dump-sql-path", default="sql/seoul_bus_172_dump.sql")
    args = ap.parse_args()

    # 폴더 준비
    raw_daily_dir = Path("data/raw/api/CardBusStatisticsServiceNew")
    raw_monthly_dir = Path("data/raw/api/CardBusTimeNew")
    ensure_dir(raw_daily_dir); ensure_dir(raw_monthly_dir)
    ensure_dir(Path("db")); ensure_dir(Path("sql"))

    # DB 준비
    conn = sqlite3.connect(args.db_path)
    conn.executescript(CREATE_TABLES_SQL)

    # 0) 노선 매핑 적재 (optional)
    upsert_dim_route_map(conn, Path(args.route_map_csv))
    conn.commit()

    # 1) 링크거리 적재
    upsert_dim_stop_link(conn, Path(args.link_csv))
    conn.commit()

    # 2) 일별 수요(승하차 합, 정류장 단위)
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    cur = start
    while cur <= end:
        yyyymmdd = cur.strftime("%Y%m%d")
        rows = fetch_parse_daily_stats(args.api_key, yyyymmdd, args.route_no, raw_daily_dir)
        if rows:
            insert_daily(conn, rows)
            conn.commit()
        cur += timedelta(days=1)

    # 3) 월별 시간대 수요(정류장·시간)
    rows_m = fetch_parse_monthly_time(args.api_key, args.month, args.route_no, raw_monthly_dir)
    if rows_m:
        insert_monthly_hourly(conn, rows_m)
        conn.commit()

    # 4) 덤프(.sql) 저장
    with open(args.dump_sql_path, "w", encoding="utf-8") as f:
        for line in conn.iterdump():
            f.write(f"{line}\n")

    conn.close()
    print(f"✅ SQLite DB: {args.db_path}")
    print(f"✅ SQL Dump  : {args.dump_sql_path}")
    print("✅ Raw XML   : data/raw/api/... 에 저장됨")

if __name__ == "__main__":
    main()