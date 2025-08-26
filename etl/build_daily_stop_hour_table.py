# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import quote
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import sqlite3
import os

def q(x):  # URL 세그먼트 안전 인코딩
    return quote(str(x), safe="")

def fetch(url: str) -> str:
    with urlopen(url) as resp:
        return resp.read().decode("utf-8")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# 환경변수 읽기 유틸
def getenv_default(name: str, default=None):
    v = os.getenv(name)
    return v if v not in (None, "") else default

# -----------------------------
# 1) tpssStationRouteTurn (대용량, 날짜만 파라미터 / route 필터는 로컬에서)
# -----------------------------
def fetch_tpss_for_date(api_key: str, yyyymmdd: str, raw_dir: Path, page_size=1000):
    base = f"http://openapi.seoul.go.kr:8088/{q(api_key)}/xml/tpssStationRouteTurn"
    start, end = 1, page_size
    rows = []
    ensure_dir(raw_dir)

    # 첫 페이지
    url = f"{base}/{q(start)}/{q(end)}/{q(yyyymmdd)}"
    xml = fetch(url)
    (raw_dir / f"{yyyymmdd}_{start}_{end}.xml").write_text(xml, encoding="utf-8")
    root = ET.fromstring(xml)
    total = int(root.findtext("list_total_count", default="0"))

    def parse_rows(root_elem):
        for r in root_elem.findall("row"):
            rec = {
                "date": r.findtext("CRTR_DD", ""),
                "rte_id": r.findtext("RTE_ID", ""),
                "stop_id": r.findtext("STOPS_ID", ""),
                "stop_seq": r.findtext("STOPS_SEQ", ""),
                "bus_opr_day": r.findtext("BUS_OPR", "0")
            }
            # 0~23시간 운행횟수
            for h in range(24):
                tag = f"BUS_OPR_{h:02d}"
                v = r.findtext(tag)
                rec[f"BUS_OPR_{h:02d}"] = int(v) if v and v.strip().isdigit() else 0
            yield rec

    rows.extend(list(parse_rows(root)))

    # 다음 페이지들
    while end < total:
        start = end + 1
        end = min(start + page_size - 1, total)
        url = f"{base}/{q(start)}/{q(end)}/{q(yyyymmdd)}"
        xml = fetch(url)
        (raw_dir / f"{yyyymmdd}_{start}_{end}.xml").write_text(xml, encoding="utf-8")
        root = ET.fromstring(xml)
        rows.extend(list(parse_rows(root)))

    df = pd.DataFrame(rows)
    # 정수/문자 정리
    if df.empty:
        return df
    df["stop_seq"] = pd.to_numeric(df["stop_seq"], errors="coerce")
    for h in range(24):
        df[f"BUS_OPR_{h:02d}"] = pd.to_numeric(df[f"BUS_OPR_{h:02d}"], errors="coerce").fillna(0).astype(int)
    return df

def melt_tpss_hourly(df_tpss: pd.DataFrame) -> pd.DataFrame:
    if df_tpss.empty:
        return df_tpss
    hour_cols = [f"BUS_OPR_{h:02d}" for h in range(24)]
    df = df_tpss.melt(
        id_vars=["date","rte_id","stop_id","stop_seq","bus_opr_day"],
        value_vars=hour_cols,
        var_name="hour_tag", value_name="운행횟수"
    )
    df["시간"] = df["hour_tag"].str[-2:].astype(int)
    df = df.drop(columns=["hour_tag"])
    # 날짜 포맷
    df["기준_날짜"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    return df[["기준_날짜","rte_id","stop_id","stop_seq","시간","운행횟수"]]

# -----------------------------
# 2) CardBusStatisticsServiceNew (일별 정류장 총 승/하차 + 정류장명)
# -----------------------------
def fetch_daily_stats_for_route(api_key: str, yyyymmdd: str, route_no: str, raw_dir: Path):
    base = f"http://openapi.seoul.go.kr:8088/{q(api_key)}/xml/CardBusStatisticsServiceNew"
    url = f"{base}/{q(1)}/{q(500)}/{q(yyyymmdd)}/{q(route_no)}/"
    xml = fetch(url)
    ensure_dir(raw_dir)
    (raw_dir / f"{yyyymmdd}_{route_no}.xml").write_text(xml, encoding="utf-8")
    root = ET.fromstring(xml)

    rows = []
    for row in root.findall("row"):
        reg_ymd = row.findtext("REG_YMD","")
        rows.append({
            "date": row.findtext("USE_YMD",""),
            "rte_id": row.findtext("RTE_ID",""),
            "route_no": row.findtext("RTE_NO",""),
            "stop_id": row.findtext("STOPS_ID",""),
            "stop_name": row.findtext("SBWY_STNS_NM",""),
            "board_total_day": int(row.findtext("GTON_TNOPE","0") or 0),
            "alight_total_day": int(row.findtext("GTOFF_TNOPE","0") or 0),
            "reg_ymd": reg_ymd
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["기준_날짜"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    return df[["기준_날짜","rte_id","route_no","stop_id","stop_name","board_total_day","alight_total_day","reg_ymd"]]

# -----------------------------
# 3) CardBusTimeNew (월별 정류장 시간대 승/하차 분포 → 가중치)
# -----------------------------
def fetch_monthly_time_for_route(api_key: str, yyyymm: str, route_no: str, raw_dir: Path):
    base = f"http://openapi.seoul.go.kr:8088/{q(api_key)}/xml/CardBusTimeNew"
    url = f"{base}/{q(1)}/{q(500)}/{q(yyyymm)}/{q(route_no)}/"
    xml = fetch(url)
    ensure_dir(raw_dir)
    (raw_dir / f"{yyyymm}_{route_no}.xml").write_text(xml, encoding="utf-8")
    root = ET.fromstring(xml)

    def get_hour(row, h, on=True):
        tags = [f"HR_{h}_GET_ON_TNOPE", f"HR_{h}_GET_ON_NOPE"] if on else [f"HR_{h}_GET_OFF_TNOPE", f"HR_{h}_GET_OFF_NOPE"]
        for t in tags:
            v = row.findtext(t)
            if v is not None:
                try:
                    return int(v)
                except:
                    return 0
        return 0

    rows = []
    for r in root.findall("row"):
        base_rec = {
            "use_ym": r.findtext("USE_YM",""),
            "route_no": r.findtext("RTE_NO",""),
            "stop_id": r.findtext("STOPS_ID",""),
            "stop_name": r.findtext("SBWY_STNS_NM",""),
            "reg_ymd": r.findtext("REG_YMD","")
        }
        for h in range(24):
            rows.append({
                **base_rec,
                "시간": h,
                "m_board": get_hour(r, h, True),
                "m_alight": get_hour(r, h, False)
            })
    df = pd.DataFrame(rows)
    return df

def build_hour_weights(df_m: pd.DataFrame) -> pd.DataFrame:
    if df_m.empty:
        return df_m
    g = df_m.groupby(["route_no","stop_id"], as_index=False)[["m_board","m_alight"]].sum().rename(columns={"m_board":"sum_board","m_alight":"sum_alight"})
    df = df_m.merge(g, on=["route_no","stop_id"], how="left")
    # 가중치(합=1). 합이 0이면 균등 분배
    df["w_board"] = df.apply(lambda r: (r["m_board"]/r["sum_board"]) if r["sum_board"]>0 else (1/24), axis=1)
    df["w_alight"] = df.apply(lambda r: (r["m_alight"]/r["sum_alight"]) if r["sum_alight"]>0 else (1/24), axis=1)
    return df[["route_no","stop_id","시간","w_board","w_alight"]]

# -----------------------------
# DB helpers (SQLite)
# -----------------------------
def connect_sqlite(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def create_dim_fact_tables_sqlite(conn):
    # Create tables with SQLite syntax
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS dim_route_map (
          route_key TEXT PRIMARY KEY,
          rte_id    TEXT NOT NULL,
          route_no  TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dim_stop_link (
          route_key TEXT NOT NULL,
          stop_id   TEXT NOT NULL,
          link_distance_m REAL,
          stop_seq  INTEGER,
          PRIMARY KEY(route_key, stop_id, stop_seq)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS fact_demand_daily_stop (
          date TEXT NOT NULL,
          rte_id TEXT,
          route_no TEXT,
          stop_id TEXT,
          stop_name TEXT,
          board_total INTEGER,
          alight_total INTEGER,
          reg_ymd TEXT,
          UNIQUE (date, route_no, stop_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS fact_demand_monthly_hourly_stop (
          use_ym TEXT NOT NULL,
          route_no TEXT,
          stop_id TEXT,
          stop_name TEXT,
          hour INTEGER,
          board_total INTEGER,
          alight_total INTEGER,
          reg_ymd TEXT,
          UNIQUE (use_ym, route_no, stop_id, hour)
        );
        """
    ]
    cur = conn.cursor()
    for stmt in ddl:
        cur.execute(stmt)
    conn.commit()

def create_flat_table_if_not_exists(conn, table: str):
    # Flat table for final output (SQLite syntax)
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table} (
      기준_날짜 TEXT NOT NULL,
      노선ID TEXT,
      정류장_ID TEXT NOT NULL,
      시간 INTEGER NOT NULL,
      승차인원 REAL,
      하차인원 REAL,
      노선번호 TEXT,
      역명 TEXT,
      정류장_순서 INTEGER,
      운행횟수 INTEGER,
      링크_구간거리_m REAL,
      UNIQUE (기준_날짜, 노선번호, 정류장_ID, 시간)
    );
    """
    cur = conn.cursor()
    cur.execute(ddl)
    conn.commit()
# -----------------------------
# 4) 링크거리 & 노선ID 매핑
# -----------------------------
def create_dim_fact_tables_mysql(conn):
    ddl = """
    CREATE TABLE IF NOT EXISTS dim_route_map (
      route_key VARCHAR(64) PRIMARY KEY,
      rte_id    VARCHAR(64) NOT NULL,
      route_no  VARCHAR(32) NOT NULL,
      INDEX idx_route_map_rte (rte_id),
      INDEX idx_route_map_no  (route_no)
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

    CREATE TABLE IF NOT EXISTS dim_stop_link (
      route_key VARCHAR(64) NOT NULL,
      stop_id   VARCHAR(64) NOT NULL,
      link_distance_m DOUBLE,
      stop_seq  INT,
      PRIMARY KEY(route_key, stop_id, stop_seq)
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

    CREATE TABLE IF NOT EXISTS fact_demand_daily_stop (
      date DATE NOT NULL,
      rte_id VARCHAR(64),
      route_no VARCHAR(32),
      stop_id VARCHAR(64),
      stop_name VARCHAR(255),
      board_total INT,
      alight_total INT,
      reg_ymd DATE,
      UNIQUE KEY uq_daily (date, route_no, stop_id),
      INDEX idx_daily_route_date (route_no, date),
      INDEX idx_daily_stop (stop_id)
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

    CREATE TABLE IF NOT EXISTS fact_demand_monthly_hourly_stop (
      use_ym CHAR(6) NOT NULL,
      route_no VARCHAR(32),
      stop_id VARCHAR(64),
      stop_name VARCHAR(255),
      hour TINYINT UNSIGNED,
      board_total INT,
      alight_total INT,
      reg_ymd DATE,
      UNIQUE KEY uq_monthly (use_ym, route_no, stop_id, hour),
      INDEX idx_monthly_route_hour (route_no, use_ym, hour),
      INDEX idx_monthly_stop (stop_id)
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
    """
    with conn.cursor() as cur:
        for stmt in ddl.split(";"):
            s = stmt.strip()
            if s:
                cur.execute(s + ";")
    conn.commit()

# SQLite upsert helpers
def upsert_dim_route_map_sqlite(conn, route_map_csv: Path):
    if not route_map_csv.exists():
        print(f"[warn] route map CSV not found: {route_map_csv} (skip)")
        return
    df = pd.read_csv(route_map_csv)
    df = df.rename(columns={"노선_ID":"route_key", "RTE_ID":"rte_id", "RTE_NO":"route_no"})
    df["route_key"] = df["route_key"].astype(str)
    df["rte_id"] = df["rte_id"].astype(str)
    df["route_no"] = df["route_no"].astype(str)
    sql = """
    INSERT OR REPLACE INTO dim_route_map (route_key, rte_id, route_no)
    VALUES (?, ?, ?)
    """
    data = list(df[["route_key","rte_id","route_no"]].itertuples(index=False, name=None))
    cur = conn.cursor()
    cur.executemany(sql, data)
    conn.commit()

def upsert_dim_stop_link_sqlite(conn, link_csv: Path):
    df = pd.read_csv(link_csv)
    df = df.rename(columns={"노선_ID":"route_key","정류장_ID":"stop_id","링크_구간거리(m)":"link_distance_m","정류장_순서":"stop_seq"})
    df["route_key"] = df["route_key"].astype(str)
    df["stop_id"] = df["stop_id"].astype(str)
    sql = """
    INSERT OR REPLACE INTO dim_stop_link (route_key, stop_id, link_distance_m, stop_seq)
    VALUES (?, ?, ?, ?)
    """
    data = list(df[["route_key","stop_id","link_distance_m","stop_seq"]].itertuples(index=False, name=None))
    cur = conn.cursor()
    cur.executemany(sql, data)
    conn.commit()

def upsert_fact_demand_daily_sqlite(conn, df_daily: pd.DataFrame):
    if df_daily is None or df_daily.empty:
        return
    tmp = df_daily.rename(columns={
        "기준_날짜":"date",
        "rte_id":"rte_id",
        "route_no":"route_no",
        "stop_id":"stop_id",
        "stop_name":"stop_name",
        "board_total_day":"board_total",
        "alight_total_day":"alight_total",
        "reg_ymd":"reg_ymd"
    }).copy()
    try:
        tmp["reg_ymd"] = pd.to_datetime(tmp["reg_ymd"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    sql = """
    INSERT OR REPLACE INTO fact_demand_daily_stop
    (date, rte_id, route_no, stop_id, stop_name, board_total, alight_total, reg_ymd)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    data = list(tmp[["date","rte_id","route_no","stop_id","stop_name","board_total","alight_total","reg_ymd"]].itertuples(index=False, name=None))
    cur = conn.cursor()
    cur.executemany(sql, data)
    conn.commit()

def upsert_fact_demand_monthly_hourly_sqlite(conn, df_monthly: pd.DataFrame):
    if df_monthly is None or df_monthly.empty:
        return
    tmp = df_monthly.rename(columns={
        "use_ym":"use_ym",
        "route_no":"route_no",
        "stop_id":"stop_id",
        "stop_name":"stop_name",
        "시간":"hour",
        "m_board":"board_total",
        "m_alight":"alight_total",
        "reg_ymd":"reg_ymd"
    }).copy()
    try:
        tmp["reg_ymd"] = pd.to_datetime(tmp["reg_ymd"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    sql = """
    INSERT OR REPLACE INTO fact_demand_monthly_hourly_stop
    (use_ym, route_no, stop_id, stop_name, hour, board_total, alight_total, reg_ymd)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    data = list(tmp[["use_ym","route_no","stop_id","stop_name","hour","board_total","alight_total","reg_ymd"]].itertuples(index=False, name=None))
    cur = conn.cursor()
    cur.executemany(sql, data)
    conn.commit()

def upsert_fact_ops_hourly_sqlite(conn, df_tpss: pd.DataFrame):
    """
    Store hourly ops data from tpss_long into fact_ops_hourly_stop table.
    """
    if df_tpss is None or df_tpss.empty:
        return
    tmp = df_tpss.rename(columns={
        "기준_날짜":"date",
        "rte_id":"rte_id",
        "stop_id":"stop_id",
        "stop_seq":"stop_seq",
        "시간":"hour",
        "운행횟수":"ops"
    }).copy()
    sql = """
    CREATE TABLE IF NOT EXISTS fact_ops_hourly_stop (
      date TEXT NOT NULL,
      rte_id TEXT,
      stop_id TEXT,
      stop_seq INTEGER,
      hour INTEGER,
      ops INTEGER,
      UNIQUE (date, rte_id, stop_id, hour)
    );
    """
    conn.execute(sql)
    data = list(tmp[["date","rte_id","stop_id","stop_seq","hour","ops"]].itertuples(index=False, name=None))
    conn.executemany("""
      INSERT OR REPLACE INTO fact_ops_hourly_stop
      (date, rte_id, stop_id, stop_seq, hour, ops)
      VALUES (?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()

def upsert_dataframe(conn, table: str, df: pd.DataFrame):
    # Column order aligned with the final output schema
    cols = [
        "기준_날짜","노선ID","정류장_ID","시간",
        "승차인원","하차인원","노선번호","역명",
        "정류장_순서","운행횟수","링크_구간거리(m)"
    ]
    # Replace NaN with None for DB insertion
    df2 = df[cols].where(pd.notnull(df[cols]), None)

    # Rename column for SQLite compatibility (링크_구간거리(m) → 링크_구간거리_m)
    insert_cols = [c if c != "링크_구간거리(m)" else "링크_구간거리_m" for c in cols]
    col_list = ", ".join(insert_cols)
    placeholders = ", ".join(["?"] * len(cols))

    # Data for upsert: rename column in DataFrame for SQLite
    df2 = df2.rename(columns={"링크_구간거리(m)": "링크_구간거리_m"})
    data = [tuple(row[c if c != "링크_구간거리(m)" else "링크_구간거리_m"] for c in cols) for _, row in df2.iterrows()]
    if not data:
        return
    sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"
    cur = conn.cursor()
    cur.executemany(sql, data)
    conn.commit()

# -----------------------------
# 4) 링크거리 & 노선ID 매핑
# -----------------------------
def load_link_distance(link_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(link_csv)
    # 기대 컬럼: 노선_ID,정류장_ID,링크_구간거리(m),정류장_순서
    df = df.rename(columns={"정류장_ID":"stop_id","링크_구간거리(m)":"link_distance_m","정류장_순서":"link_stop_seq"})
    df["stop_id"] = df["stop_id"].astype(str)
    df["노선_ID"] = df["노선_ID"].astype(str)
    return df

def load_route_map(route_map_csv: Path) -> pd.DataFrame:
    # 기대 컬럼: 노선_ID, RTE_ID, RTE_NO
    if not route_map_csv.exists():
        print(f"[warn] route map CSV not found: {route_map_csv} (skip; 링크거리 매칭은 노선_ID 직접 지정 필요)")
        return pd.DataFrame()
    df = pd.read_csv(route_map_csv)
    df["노선_ID"] = df["노선_ID"].astype(str)
    df["RTE_ID"] = df["RTE_ID"].astype(str)
    df["RTE_NO"] = df["RTE_NO"].astype(str)
    return df

# -----------------------------
# 메인 조립
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-key", required=True, help="CardBusStatistics/Time API Key")
    ap.add_argument("--tpss-key", required=True, help="tpssStationRouteTurn API Key")
    ap.add_argument("--route-no", required=True, help="노선번호 (예: 172)")
    ap.add_argument("--rte-id", required=True, help="RTE_ID (예: 11110028)")
    ap.add_argument("--date", required=True, help="YYYYMMDD (예: 20250601)")
    ap.add_argument("--month", required=True, help="YYYYMM (예: 202506)")
    ap.add_argument("--link-csv", required=True, help="노선_ID,정류장_ID,링크_구간거리(m),정류장_순서 CSV")
    ap.add_argument("--route-map-csv", required=True, help="노선_ID,RTE_ID,RTE_NO 매핑 CSV")
    ap.add_argument("--db-path", required=True, help="SQLite DB file path")
    ap.add_argument("--db-table", default="daily_stop_hour", help="Target table name (default: daily_stop_hour)")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    db_path = args.db_path
    db_table = args.db_table

    d_yyyy_mm_dd = datetime.strptime(args.date, "%Y%m%d").strftime("%Y-%m-%d")

    # 1) TPS스(대용량) 수집 or 로드 from DB
    conn = connect_sqlite(db_path)
    # Ensure fact_ops_hourly_stop exists
    conn.execute("""
    CREATE TABLE IF NOT EXISTS fact_ops_hourly_stop (
      date TEXT NOT NULL,
      rte_id TEXT,
      stop_id TEXT,
      stop_seq INTEGER,
      hour INTEGER,
      ops INTEGER,
      UNIQUE (date, rte_id, stop_id, hour)
    );
    """)
    # Check if data for this date+rte_id exists
    cur = conn.cursor()
    cur.execute("SELECT COUNT(1) FROM fact_ops_hourly_stop WHERE date=? AND rte_id=?", (datetime.strptime(args.date, "%Y%m%d").strftime("%Y-%m-%d"), str(args.rte_id)))
    tpss_exists = cur.fetchone()[0] > 0
    if tpss_exists:
        # Load from DB
        tpss_long = pd.read_sql_query(
            "SELECT date as 기준_날짜, rte_id, stop_id, stop_seq, hour as 시간, ops as 운행횟수 FROM fact_ops_hourly_stop WHERE date=? AND rte_id=?",
            conn,
            params=(datetime.strptime(args.date, "%Y%m%d").strftime("%Y-%m-%d"), str(args.rte_id))
        )
    else:
        # Fetch from API, process, and store in DB
        tpss_raw = fetch_tpss_for_date(args.tpss_key, args.date, Path("data/raw/api/tpss"))
        tpss_raw = tpss_raw[tpss_raw["rte_id"].astype(str) == str(args.rte_id)]
        tpss_long = melt_tpss_hourly(tpss_raw)  # [기준_날짜, rte_id, stop_id, stop_seq, 시간, 운행횟수]
        upsert_fact_ops_hourly_sqlite(conn, tpss_long)

    # 2) 일별 정류장 총 승/하차 + 정류장명
    daily_stats = fetch_daily_stats_for_route(args.stats_key, args.date, args.route_no, Path("data/raw/api/CardBusStatisticsServiceNew"))
    # 3) 월별 시간대 분포 → 가중치
    monthly = fetch_monthly_time_for_route(args.stats_key, args.month, args.route_no, Path("data/raw/api/CardBusTimeNew"))
    weights = build_hour_weights(monthly)  # [route_no, stop_id, 시간, w_board, w_alight]

    # 4) 링크거리 & 라우트 매핑
    link_df = load_link_distance(Path(args.link_csv))  # [노선_ID, stop_id, link_distance_m, link_stop_seq]
    route_map = load_route_map(Path(args.route_map_csv))  # [노선_ID, RTE_ID, RTE_NO]

    # 이 노선의 노선_ID 찾기
    route_key = None
    if not route_map.empty:
        m = route_map[(route_map["RTE_ID"]==str(args.rte_id)) & (route_map["RTE_NO"]==str(args.route_no))]
        if not m.empty:
            route_key = m.iloc[0]["노선_ID"]
    if route_key is None:
        print("[warn] route_key(노선_ID)를 route_map에서 찾지 못했어요. 링크거리 조인은 skip(또는 직접 --route-id 추가 구현).")

    # 5) 조립: fact_ops_hourly_stop(시간대 운행횟수) + 일별 총량 + 월 가중치
    #    - 시간 단위에 승/하차 분배
    #    - 정류장명 붙이기
    if tpss_long.empty:
        print("tpss 데이터가 비었습니다. 종료.")
        conn.close()
        return

    # 정류장명(일별) 붙이기
    name_map = daily_stats[["기준_날짜","stop_id","stop_name","route_no"]].drop_duplicates() if not daily_stats.empty else pd.DataFrame(columns=["기준_날짜","stop_id","stop_name","route_no"])
    df = tpss_long.merge(name_map, on=["기준_날짜","stop_id"], how="left")

    # 일별 총량 붙이기
    daily_map = daily_stats[["기준_날짜","stop_id","board_total_day","alight_total_day"]].drop_duplicates() if not daily_stats.empty else None
    if daily_map is not None and not daily_map.empty:
        df = df.merge(daily_map, on=["기준_날짜","stop_id"], how="left")
    else:
        df["board_total_day"] = 0
        df["alight_total_day"] = 0

    # 월 가중치 붙이기 (route_no는 문자열로 정규화)
    df["route_no"] = str(args.route_no)
    weights["route_no"] = weights["route_no"].astype(str)
    df = df.merge(weights, on=["route_no","stop_id","시간"], how="left")
    # 가중치가 없는 경우 균등 분배
    df["w_board"] = df["w_board"].fillna(1/24)
    df["w_alight"] = df["w_alight"].fillna(1/24)

    # 시간대 승/하차 추정
    df["승차인원"] = df["board_total_day"].fillna(0) * df["w_board"]
    df["하차인원"] = df["alight_total_day"].fillna(0) * df["w_alight"]

    # 링크거리 붙이기 (가능하면)
    if route_key is not None:
        link_sub = link_df[link_df["노선_ID"] == str(route_key)].copy()
        # 우선 stop_id로 붙이고, 없으면 seq로 보조 매칭
        df = df.merge(link_sub[["stop_id","link_distance_m"]], on="stop_id", how="left")
        # 보조: stop_seq == link_stop_seq로 매칭 (아직 빈 곳만)
        if "link_distance_m" in df.columns:
            missing = df["link_distance_m"].isna()
            if missing.any():
                df2 = df[missing].merge(link_sub[["link_stop_seq","link_distance_m"]], left_on="stop_seq", right_on="link_stop_seq", how="left")
                df.loc[missing, "link_distance_m"] = df2["link_distance_m"].values
        # 노선ID 컬럼 추가
        df["노선ID"] = str(route_key)
    else:
        df["link_distance_m"] = None
        df["노선ID"] = None

    # 최종 컬럼 구성/정렬
    df = df.rename(columns={
        "stop_id":"정류장_ID",
        "stop_name":"역명",
        "stop_seq":"정류장_순서",
        "route_no":"노선번호",
        "운행횟수":"운행횟수",
        "link_distance_m":"링크_구간거리(m)"
    })

    out_cols = ["기준_날짜","노선ID","정류장_ID","시간","승차인원","하차인원","노선번호","역명","정류장_순서","운행횟수","링크_구간거리(m)"]
    # 운행횟수 없으면 0으로
    if "운행횟수" not in df.columns:
        df["운행횟수"] = 0
    # 타입 정리
    df["정류장_순서"] = pd.to_numeric(df["정류장_순서"], errors="coerce").astype("Int64")
    df = df[out_cols].sort_values(["기준_날짜","시간","정류장_순서","정류장_ID"])

    # ---- Save to DB (normalized + flat) ----
    create_dim_fact_tables_sqlite(conn)
    upsert_dim_route_map_sqlite(conn, Path(args.route_map_csv))
    upsert_dim_stop_link_sqlite(conn, Path(args.link_csv))
    upsert_fact_demand_daily_sqlite(conn, daily_stats)
    upsert_fact_demand_monthly_hourly_sqlite(conn, monthly)
    create_flat_table_if_not_exists(conn, db_table)
    upsert_dataframe(conn, db_table, df)
    conn.close()
    print(f"✅ upserted {len(df):,} rows into `{db_table}` and refreshed normalized tables in SQLite DB {db_path}")

    # ---- Save to CSV (optional) ----
    if args.out_csv:
        ensure_dir(Path(args.out_csv).parent)
        df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        print(f"✅ saved CSV: {args.out_csv}")

if __name__ == "__main__":
    main()