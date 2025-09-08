import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import quote
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import sqlite3
import os
import io
import sys
try:
    import boto3
except Exception:
    boto3 = None

def q(x):  
    return quote(str(x), safe="")

def fetch(url: str) -> str:
    with urlopen(url) as resp:
        return resp.read().decode("utf-8")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def getenv_default(name: str, default=None):
    v = os.getenv(name)
    return v if v not in (None, "") else default

def ensure_audit_table_sqlite(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS etl_run_audit (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      job_name TEXT NOT NULL,
      target_date TEXT NOT NULL,  -- YYYYMMDD
      status TEXT NOT NULL,       -- OK / SKIPPED_SAME_DATE / NO_DATA / ERROR
      message TEXT,
      created_at TEXT NOT NULL    -- ISO timestamp
    );
    """)
    conn.commit()

def get_last_target_date_sqlite(conn, job_name: str):
    cur = conn.cursor()
    cur.execute("""
      SELECT target_date
      FROM etl_run_audit
      WHERE job_name = ?
      ORDER BY id DESC
      LIMIT 1
    """, (job_name,))
    row = cur.fetchone()
    return row[0] if row else None

def append_audit_sqlite(conn, job_name: str, target_date: str, status: str, message: str = None):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
      INSERT INTO etl_run_audit (job_name, target_date, status, message, created_at)
      VALUES (?, ?, ?, ?, ?)
    """, (job_name, target_date, status, message, ts))
    conn.commit()

# ---- S3 Upload Helper ----
def s3_upload_bytes(bucket: str, key: str, data: bytes, content_type: str = "text/csv"):
    """
    Upload raw bytes to S3. Requires that boto3 is available and credentials/role are configured.
    """
    if not boto3:
        print("[warn] boto3 not available; skip S3 upload.")
        return
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    print(f"✅ uploaded to s3://{bucket}/{key}")

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
# TPSS CSV 로드 헬퍼
# -----------------------------
def load_tpss_from_csv(csv_path, route_no: str, rte_id: str, yyyymmdd: str, route_map_csv=None) -> pd.DataFrame:
    """
    로컬 CSV(정류장별/시간대별 버스 운행횟수 피벗 형태)를 읽어
    [기준_날짜, rte_id, stop_id, stop_seq, 시간, 운행횟수] Long 형태로 변환.
    CSV 기대 컬럼 예:
      - 기준_날짜 (YYYYMMDD)
      - 노선_ID
      - 정류장_ID
      - 버스운행횟수_일
      - 버스운행횟수_00시 ... 버스운행횟수_23시
      - 정류장_순서
    """
    df = pd.read_csv(csv_path, dtype=str)
    # 날짜 필터
    df = df[df["기준_날짜"] == yyyymmdd].copy()
    if df.empty:
        return pd.DataFrame(columns=["기준_날짜","rte_id","stop_id","stop_seq","시간","운행횟수"])

    # route_key(노선_ID) 매핑: route_map_csv에서 (RTE_NO, RTE_ID) → 노선_ID
    route_key = None
    if route_map_csv:
        try:
            rm = pd.read_csv(route_map_csv, dtype=str)
            hit = rm[(rm["RTE_NO"].astype(str) == str(route_no)) & (rm["RTE_ID"].astype(str) == str(rte_id))]
            if not hit.empty:
                route_key = hit.iloc[0]["노선_ID"]
        except FileNotFoundError:
            route_key = None
        except Exception:
            route_key = None

    if route_key is not None:
        df = df[df["노선_ID"].astype(str) == str(route_key)].copy()

    # 시간 컬럼 수집
    hour_cols = [c for c in df.columns if c.startswith("버스운행횟수_") and c.endswith("시")]
    # 안전하게 00~23시만 정렬
    def _hkey(c): 
        try:
            return int(c.split("_")[-1].replace("시",""))
        except:
            return 99
    hour_cols = sorted(hour_cols, key=_hkey)

    # 결측/빈칸 → 0
    for c in hour_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # melt
    long_df = df.melt(
        id_vars=["기준_날짜","정류장_ID","정류장_순서"],
        value_vars=hour_cols,
        var_name="hour_tag", value_name="운행횟수"
    )
    long_df["시간"] = long_df["hour_tag"].str.extract(r"(\d+)").astype(int)
    long_df = long_df.drop(columns=["hour_tag"])

    # 형식 맞추기
    long_df = long_df.rename(columns={"정류장_ID":"stop_id","정류장_순서":"stop_seq"})
    long_df["rte_id"] = str(rte_id)
    # 날짜 포맷 변경
    long_df["기준_날짜"] = pd.to_datetime(long_df["기준_날짜"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

    # 열 정리/순서
    long_df = long_df[["기준_날짜","rte_id","stop_id","stop_seq","시간","운행횟수"]].copy()
    # 타입 정리
    long_df["stop_seq"] = pd.to_numeric(long_df["stop_seq"], errors="coerce")
    long_df["운행횟수"] = pd.to_numeric(long_df["운행횟수"], errors="coerce").fillna(0).astype(int)
    return long_df

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

def upsert_dim_stop_link_from_master_sqlite(conn, stop_master_df: pd.DataFrame, route_key: str | None):
    if stop_master_df is None or stop_master_df.empty:
        return
    df = stop_master_df.copy()
    # 기대 컬럼: route_key(없으면 채움), stop_id, link_distance_m, link_stop_seq
    if "stop_id" not in df.columns or "link_stop_seq" not in df.columns:
        return
    if "link_distance_m" not in df.columns:
        return
    if "route_key" not in df.columns:
        df["route_key"] = route_key if route_key is not None else None
    df = df.dropna(subset=["stop_id","link_stop_seq"])
    tuples = list(df[["route_key","stop_id","link_distance_m","link_stop_seq"]]
                    .rename(columns={"link_stop_seq":"stop_seq"})
                    .itertuples(index=False, name=None))
    if not tuples:
        return
    sql = """
    INSERT OR REPLACE INTO dim_stop_link (route_key, stop_id, link_distance_m, stop_seq)
    VALUES (?, ?, ?, ?)
    """
    conn.executemany(sql, tuples)
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

def load_stop_master(stop_master_csv: Path, route_no: str, rte_id: str, yyyymmdd: str) -> pd.DataFrame:
    """
    유연 로더: 열 이름이 달라도 최대한 매칭.
    반환 컬럼: ['route_key','stop_id','link_distance_m','link_stop_seq']
    날짜 컬럼이 있으면 YYYYMMDD(예: 20250801)로 필터링.
    """
    if not stop_master_csv or not Path(stop_master_csv).exists():
        return pd.DataFrame(columns=["route_key","stop_id","link_distance_m","link_stop_seq"])

    df = pd.read_csv(stop_master_csv, dtype=str)

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    col_route_key = pick("노선_ID","ROUTE_KEY","ROUTE_ID")
    col_rte_id    = pick("RTE_ID","노선아이디")
    col_route_no  = pick("RTE_NO","노선번호","ROUTE_NO")
    col_stop_id   = pick("정류장_ID","STOPS_ID","STOP_ID")
    col_stop_seq  = pick("정류장_순서","STOPS_SEQ","STOP_SEQ")
    col_dist      = pick("링크_구간거리(m)","LINK_DISTANCE_M","링크거리","LINK_DIST_M")
    col_date      = pick("CRTR_DD","기준_날짜","DATE")

    if col_stop_id is None or col_stop_seq is None:
        return pd.DataFrame(columns=["route_key","stop_id","link_distance_m","link_stop_seq"])

    # 날짜 필터(있으면)
    if col_date is not None:
        d = df[col_date].astype(str).str.replace(r"[^0-9]","", regex=True)
        df = df[d == str(yyyymmdd)]

    # 노선 필터(가능하면)
    if col_rte_id is not None:
        df = df[df[col_rte_id].astype(str) == str(rte_id)]
    elif col_route_no is not None:
        df = df[df[col_route_no].astype(str) == str(route_no)]

    if df.empty:
        return pd.DataFrame(columns=["route_key","stop_id","link_distance_m","link_stop_seq"])

    out = pd.DataFrame()
    out["route_key"] = df[col_route_key].astype(str) if col_route_key else None
    out["stop_id"] = df[col_stop_id].astype(str)
    out["link_stop_seq"] = pd.to_numeric(df[col_stop_seq], errors="coerce")
    out["link_distance_m"] = pd.to_numeric(df[col_dist], errors="coerce") if col_dist else pd.NA
    out = out.dropna(subset=["stop_id","link_stop_seq"])
    return out

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
    ap.add_argument("--link-csv", default=None,
                help="(선택) 노선_ID,정류장_ID,링크_구간거리(m),정류장_순서 CSV. stop-master에 링크거리가 있으면 생략 가능")
    ap.add_argument("--stop-master-csv", default=None,
                help="(선택) 노선 정류장 마스터 CSV 경로 (노선_ID, 정류장_ID, 정류장_순서, [링크_구간거리(m)], [CRTR_DD] 포함 가능)")
    ap.add_argument("--route-map-csv", default=None, help="(선택) 노선_ID=RTE_ID 가정 시 불필요. 제공 시 노선_ID,RTE_ID,RTE_NO 매핑 CSV")
    ap.add_argument("--db-path", required=True, help="SQLite DB file path")
    ap.add_argument("--db-table", default="daily_stop_hour", help="Target table name (default: daily_stop_hour)")
    ap.add_argument("--tpss-csv", default=None, help="(선택) tpssStationRouteTurn을 대신할 로컬 CSV 경로")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--s3-bucket", default=getenv_default("S3_BUCKET"), help="(선택) 업로드할 S3 버킷명. 환경변수 S3_BUCKET 사용 가능")
    ap.add_argument("--s3-prefix", default=getenv_default("S3_PREFIX", "processed"), help="(선택) S3 키 prefix (기본: processed)")
    args = ap.parse_args()

    db_path = args.db_path
    db_table = args.db_table

    # ---- Audit: skip if target date equals previous run's target ----
    job_name = getenv_default("JOB_NAME", "build_daily_stop_hour_table")
    # We'll open a lightweight connection first only for audit check
    conn_audit = connect_sqlite(db_path)
    ensure_audit_table_sqlite(conn_audit)
    last_target = get_last_target_date_sqlite(conn_audit, job_name)
    if last_target is not None and str(last_target) == str(args.date):
        msg = f"[error] target date {args.date} is identical to previous run; skip execution."
        print(msg, file=sys.stderr)
        append_audit_sqlite(conn_audit, job_name, str(args.date), "SKIPPED_SAME_DATE", msg)
        conn_audit.close()
        return
    # keep this audit connection to append final status later if you prefer, or close it now:
    conn_audit.close()

    d_yyyy_mm_dd = datetime.strptime(args.date, "%Y%m%d").strftime("%Y-%m-%d")

    # 노선_ID == RTE_ID 가정 (서울시 내 통일)
    route_key = str(args.rte_id)
    route_map = pd.DataFrame()  # 매핑 CSV 없이 진행
    # Prefer stop-master for link/sequence if provided (date-filtered)
    stop_master_df = load_stop_master(Path(args.stop_master_csv), args.route_no, args.rte_id, args.date) if args.stop_master_csv else pd.DataFrame()

    # 1) TPS스(대용량) 수집/로드
    conn = connect_sqlite(db_path)
    # ensure audit table exists on the main connection too (so we can log outcomes)
    ensure_audit_table_sqlite(conn)
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

    yyyy_mm_dd_str = datetime.strptime(args.date, "%Y%m%d").strftime("%Y-%m-%d")

    if args.tpss_csv:
        # 🚀 빠른 경로: 로컬 CSV 사용
        tpss_long = load_tpss_from_csv(args.tpss_csv, args.route_no, args.rte_id, args.date, args.route_map_csv)
        if tpss_long.empty:
            print("[warn] TPSS CSV에서 해당 날짜/노선 데이터를 찾지 못했습니다.")
        else:
            upsert_fact_ops_hourly_sqlite(conn, tpss_long)
    else:
        # 기존 경로: API 호출 → DB 캐시
        cur = conn.cursor()
        cur.execute("SELECT COUNT(1) FROM fact_ops_hourly_stop WHERE date=? AND rte_id=?", (yyyy_mm_dd_str, str(args.rte_id)))
        tpss_exists = cur.fetchone()[0] > 0
        if tpss_exists:
            # Load from DB
            tpss_long = pd.read_sql_query(
                "SELECT date as 기준_날짜, rte_id, stop_id, stop_seq, hour as 시간, ops as 운행횟수 FROM fact_ops_hourly_stop WHERE date=? AND rte_id=?",
                conn, params=(yyyy_mm_dd_str, str(args.rte_id))
            )
        else:
            # Fetch from API, process, and store in DB
            tpss_raw = fetch_tpss_for_date(args.tpss_key, args.date, Path('data/raw/api/tpss'))
            tpss_raw = tpss_raw[tpss_raw['rte_id'].astype(str) == str(args.rte_id)]
            tpss_long = melt_tpss_hourly(tpss_raw)
            upsert_fact_ops_hourly_sqlite(conn, tpss_long)

    # ---- Hard filter to the single target route (rte_id) ----
    if 'tpss_long' in locals() and tpss_long is not None and not tpss_long.empty:
        tpss_long = tpss_long[tpss_long['rte_id'].astype(str) == str(args.rte_id)]

    # 2) 일별 정류장 총 승/하차 + 정류장명
    daily_stats = fetch_daily_stats_for_route(args.stats_key, args.date, args.route_no, Path("data/raw/api/CardBusStatisticsServiceNew"))
    # 3) 월별 시간대 분포 → 가중치
    monthly = fetch_monthly_time_for_route(args.stats_key, args.month, args.route_no, Path("data/raw/api/CardBusTimeNew"))
    weights = build_hour_weights(monthly)  # [route_no, stop_id, 시간, w_board, w_alight]

    # 4) 링크거리 & 라우트 매핑 (stop-master 우선, 없으면 link-csv)
    if stop_master_df is not None and not stop_master_df.empty:
        # stop_master_df columns: route_key, stop_id, link_distance_m, link_stop_seq
        link_df = stop_master_df.copy()
    else:
        # fallback to legacy link CSV (노선_ID, 정류장_ID, 링크_구간거리(m), 정류장_순서)
        link_df = load_link_distance(Path(args.link_csv))  # returns: [노선_ID, stop_id, link_distance_m, link_stop_seq]

    # route_key는 RTE_ID와 동일하게 사용
    # route_key = str(args.rte_id)

    # 5) 조립: fact_ops_hourly_stop(시간대 운행횟수) + 일별 총량 + 월 가중치
    #    - 시간 단위에 승/하차 분배
    #    - 정류장명 붙이기
    if tpss_long.empty:
        print("tpss 데이터가 비었습니다. 종료.")
        try:
            append_audit_sqlite(conn, getenv_default("JOB_NAME", "build_daily_stop_hour_table"), str(args.date), "NO_DATA", "tpss empty")
        except Exception:
            pass
        conn.close()
        return

    # 정류장명(일별) 붙이기
    name_map = (
        daily_stats[["기준_날짜","stop_id","stop_name","route_no"]].drop_duplicates()
        if not daily_stats.empty
        else pd.DataFrame(columns=["기준_날짜","stop_id","stop_name","route_no"])
    )
    df = tpss_long.merge(name_map, on=["기준_날짜","stop_id"], how="left")

    # ---- Hard filter to single route: route_no & rte_id ----
    df["route_no"] = str(args.route_no)  # 보정
    df = df[(df["rte_id"].astype(str) == str(args.rte_id)) & (df["route_no"].astype(str) == str(args.route_no))]

    # 일별 총량 붙이기
    daily_map = daily_stats[["기준_날짜","stop_id","board_total_day","alight_total_day"]].drop_duplicates() if not daily_stats.empty else None
    if daily_map is not None and not daily_map.empty:
        df = df.merge(daily_map, on=["기준_날짜","stop_id"], how="left")
    else:
        df["board_total_day"] = 0
        df["alight_total_day"] = 0

    # --- 월 가중치 준비 (빈 경우/route_no 누락 시 균등 가중치 생성) ---
    weights = build_hour_weights(monthly)  # 기대 스키마: [route_no, stop_id, 시간, w_board, w_alight]

    # 월 가중치가 비었거나, route_no 컬럼이 없거나, 키가 맞지 않으면 균등(1/24)로 생성
    if (weights is None) or weights.empty or ("stop_id" not in weights.columns) or ("시간" not in weights.columns):
        # tpss_long에서 사용되는 정류장과 0~23시간을 기준으로 균등 가중치 구성
        stops = sorted(tpss_long["stop_id"].astype(str).unique().tolist()) if not tpss_long.empty else []
        hours = list(range(24))
        weights = pd.DataFrame(
            [(str(args.route_no), s, h, 1/24, 1/24) for s in stops for h in hours],
            columns=["route_no","stop_id","시간","w_board","w_alight"]
        )
    else:
        # 타입 정규화: 존재하는 컬럼만 안전하게 캐스팅
        if "route_no" in weights.columns:
            weights["route_no"] = weights["route_no"].astype(str)
        weights["stop_id"] = weights["stop_id"].astype(str)

    # 월 가중치 붙이기 (route_no가 있으면 route_no도 키로, 없으면 stop_id+시간만)
    df["route_no"] = str(args.route_no)
    if "route_no" in weights.columns:
        df = df.merge(weights, on=["route_no","stop_id","시간"], how="left")
    else:
        df = df.merge(weights[["stop_id","시간","w_board","w_alight"]], on=["stop_id","시간"], how="left")

    # 가중치가 없는 경우(merge 후 결측)도 균등 분배
    df["w_board"] = df["w_board"].fillna(1/24)
    df["w_alight"] = df["w_alight"].fillna(1/24)

    # 시간대 승/하차 추정
    df["승차인원"] = df["board_total_day"].fillna(0) * df["w_board"]
    df["하차인원"] = df["alight_total_day"].fillna(0) * df["w_alight"]

    # 링크거리 붙이기 (가능하면)
    if route_key is not None:
        # route_key 컬럼명 차이를 흡수
        if "route_key" in link_df.columns:
            link_sub = link_df[link_df["route_key"].astype(str) == str(route_key)].copy()
        elif "노선_ID" in link_df.columns:
            link_sub = link_df[link_df["노선_ID"].astype(str) == str(route_key)].copy()
        else:
            link_sub = link_df.copy()

        # 우선 stop_id로 붙이고, 없으면 seq로 보조 매칭
        if "stop_id" in link_sub.columns:
            df = df.merge(link_sub[["stop_id","link_distance_m"]], on="stop_id", how="left")
        # 보조: stop_seq == link_stop_seq로 매칭 (아직 빈 곳만)
        # link_distance_m 컬럼이 없다면 만들어 둔다
        if "link_distance_m" not in df.columns:
            df["link_distance_m"] = pd.NA
        missing = df["link_distance_m"].isna()
        if missing.any() and "link_stop_seq" in link_sub.columns:
            # 안전한 매핑: link_stop_seq → link_distance_m (중복은 평균 처리)
            tmp_map = (
                link_sub[["link_stop_seq", "link_distance_m"]]
                .dropna(subset=["link_stop_seq"])
                .copy()
            )
            # 타입 정규화
            tmp_map["link_stop_seq"] = pd.to_numeric(tmp_map["link_stop_seq"], errors="coerce")
            # 평균값으로 중복 축약
            seq2dist = tmp_map.groupby("link_stop_seq")["link_distance_m"].mean()
            # 결측 대상에만 매핑
            df.loc[missing, "link_distance_m"] = df.loc[missing, "stop_seq"].map(seq2dist)

        # 노선ID 컬럼 추가
        df["노선ID"] = str(route_key)
    else:
        # route_key를 찾지 못한 경우라도 stop_id 기준으로만 시도
        if "stop_id" in link_df.columns and "link_distance_m" in link_df.columns:
            df = df.merge(link_df[["stop_id","link_distance_m"]], on="stop_id", how="left")
        if "link_distance_m" not in df.columns:
            df["link_distance_m"] = pd.NA
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
    # route_map CSV가 없더라도 노선_ID == RTE_ID 가정으로 dim_route_map에 직접 upsert
    if args.route_map_csv and Path(args.route_map_csv).exists():
        upsert_dim_route_map_sqlite(conn, Path(args.route_map_csv))
    else:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS dim_route_map (
          route_key TEXT PRIMARY KEY,
          rte_id    TEXT NOT NULL,
          route_no  TEXT NOT NULL
        );
        """)
        conn.execute("INSERT OR REPLACE INTO dim_route_map (route_key, rte_id, route_no) VALUES (?, ?, ?)",
                     (str(args.rte_id), str(args.rte_id), str(args.route_no)))
        conn.commit()
    # Prefer stop-master for link/sequence if available; otherwise fallback to --link-csv
    if stop_master_df is not None and not stop_master_df.empty:
        upsert_dim_stop_link_from_master_sqlite(conn, stop_master_df, route_key)
    elif args.link_csv:
        upsert_dim_stop_link_sqlite(conn, Path(args.link_csv))
    upsert_fact_demand_daily_sqlite(conn, daily_stats)
    upsert_fact_demand_monthly_hourly_sqlite(conn, monthly)
    create_flat_table_if_not_exists(conn, db_table)
    upsert_dataframe(conn, db_table, df)
    affected = len(df)
    conn.close()

    # ---- Upload to S3 (optional) ----
    if affected > 0 and args.s3_bucket:
        # Key: {prefix}/daily_stop_hour/route={route_no}/date={date}.csv
        prefix = (args.s3_prefix or "processed").rstrip("/")
        s3_key = f"{prefix}/daily_stop_hour/route={args.route_no}/date={args.date}.csv"
        buf = io.StringIO()
        df.to_csv(buf, index=False, encoding="utf-8-sig")
        s3_upload_bytes(args.s3_bucket, s3_key, buf.getvalue().encode("utf-8-sig"), content_type="text/csv; charset=utf-8")

    try:
        if affected > 0:
            append_audit_sqlite(conn, getenv_default("JOB_NAME", "build_daily_stop_hour_table"), str(args.date), "OK", f"rows={affected}")
        else:
            append_audit_sqlite(conn, getenv_default("JOB_NAME", "build_daily_stop_hour_table"), str(args.date), "NO_DATA", "no rows affected")
    except Exception:
        pass
    if affected > 0:
        print(f"✅ upserted {affected:,} rows into `{db_table}` and refreshed normalized tables in SQLite DB {db_path}")
    else:
        print("⚠️ 생성된 행이 없습니다. 입력 데이터/필터(날짜, 노선)를 확인하세요.")

    # ---- Save to CSV (optional) ----
    if args.out_csv:
        ensure_dir(Path(args.out_csv).parent)
        df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        print(f"✅ saved CSV: {args.out_csv}")

if __name__ == "__main__":
    main()