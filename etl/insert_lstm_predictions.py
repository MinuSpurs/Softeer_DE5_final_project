import argparse
import os
import sqlite3
from pathlib import Path

import pandas as pd

try:
    import boto3  
except Exception:  
    boto3 = None

try:
    import pymysql 
except Exception:  
    pymysql = None


TABLE_NAME = "fact_lstm_predictions"

CREATE_SQLITE = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    기준_날짜 TEXT,
    시간 INTEGER,
    정류장_ID TEXT,
    정류장_순서 INTEGER,
    역명 TEXT,
    통과버스수 INTEGER,
    예측_승차_총 REAL,
    예측_하차_총 REAL,
    버스당_승차 REAL,
    버스당_하차 REAL,
    버스당_onboard_추정 REAL,
    PRIMARY KEY (기준_날짜, 시간, 정류장_ID)
);
"""

CREATE_MYSQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    `기준_날짜` DATE NOT NULL,
    `시간` TINYINT NOT NULL,
    `정류장_ID` VARCHAR(32) NOT NULL,
    `정류장_순서` INT NULL,
    `역명` VARCHAR(255) NULL,
    `통과버스수` INT NULL,
    `예측_승차_총` DOUBLE NULL,
    `예측_하차_총` DOUBLE NULL,
    `버스당_승차` DOUBLE NULL,
    `버스당_하차` DOUBLE NULL,
    `버스당_onboard_추정` DOUBLE NULL,
    PRIMARY KEY (`기준_날짜`,`시간`,`정류장_ID`),
    KEY idx_stop_hour (`정류장_ID`,`시간`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
"""

UPSERT_SQLITE = f"""
INSERT INTO {TABLE_NAME} (
    기준_날짜, 시간, 정류장_ID, 정류장_순서, 역명, 통과버스수,
    예측_승차_총, 예측_하차_총, 버스당_승차, 버스당_하차, 버스당_onboard_추정
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(기준_날짜, 시간, 정류장_ID) DO UPDATE SET
    정류장_순서=excluded.정류장_순서,
    역명=excluded.역명,
    통과버스수=excluded.통과버스수,
    예측_승차_총=excluded.예측_승차_총,
    예측_하차_총=excluded.예측_하차_총,
    버스당_승차=excluded.버스당_승차,
    버스당_하차=excluded.버스당_하차,
    버스당_onboard_추정=excluded.버스당_onboard_추정
;
"""

UPSERT_MYSQL = f"""
INSERT INTO {TABLE_NAME} (
    `기준_날짜`, `시간`, `정류장_ID`, `정류장_순서`, `역명`, `통과버스수`,
    `예측_승차_총`, `예측_하차_총`, `버스당_승차`, `버스당_하차`, `버스당_onboard_추정`
) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON DUPLICATE KEY UPDATE
    `정류장_순서`=VALUES(`정류장_순서`),
    `역명`=VALUES(`역명`),
    `통과버스수`=VALUES(`통과버스수`),
    `예측_승차_총`=VALUES(`예측_승차_총`),
    `예측_하차_총`=VALUES(`예측_하차_총`),
    `버스당_승차`=VALUES(`버스당_승차`),
    `버스당_하차`=VALUES(`버스당_하차`),
    `버스당_onboard_추정`=VALUES(`버스당_onboard_추정`);
"""


def coerce_and_align_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    모델 출력 CSV 스키마를 DB 스키마에 맞게 정렬/형변환.
    허용 가능한 대체 컬럼명이 오면 rename.
    """
    rename_candidates = {
        # 왼쪽: 요구 스키마, 오른쪽: 허용 대체명 리스트
        "기준_날짜": ["기준날짜", "date", "DATE"],
        "시간": ["hour", "HOUR"],
        "정류장_ID": ["정류장ID", "stop_id", "STOP_ID"],
        "정류장_순서": ["정류장순서", "stop_seq", "STOP_SEQ"],
        "역명": ["정류장명", "stop_name", "STOP_NAME"],
        "통과버스수": ["운행횟수", "departures", "통과대수"],
        "예측_승차_총": ["예측_승차인원", "pred_board_total", "yhat_board_total"],
        "예측_하차_총": ["예측_하차인원", "pred_alight_total", "yhat_alight_total"],
        "버스당_승차": ["예측_버스당_승차", "per_bus_board"],
        "버스당_하차": ["예측_버스당_하차", "per_bus_alight"],
        "버스당_onboard_추정": ["버스당_onboard", "per_bus_onboard"],
    }

    # rename 자동화
    rename_map = {}
    for need, cands in rename_candidates.items():
        if need in df.columns:
            continue
        for c in cands:
            if c in df.columns:
                rename_map[c] = need
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    required = [
        "기준_날짜", "시간", "정류장_ID", "정류장_순서", "역명", "통과버스수",
        "예측_승차_총", "예측_하차_총", "버스당_승차", "버스당_하차", "버스당_onboard_추정"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    # 타입 정제
    df = df.copy()
    df["기준_날짜"] = pd.to_datetime(df["기준_날짜"]).dt.strftime("%Y-%m-%d")
    df["시간"] = pd.to_numeric(df["시간"], errors="coerce").fillna(0).astype(int)
    df["정류장_ID"] = df["정류장_ID"].astype(str)
    df["정류장_순서"] = pd.to_numeric(df["정류장_순서"], errors="coerce").fillna(0).astype(int)
    for col in ["통과버스수", "예측_승차_총", "예측_하차_총", "버스당_승차", "버스당_하차", "버스당_onboard_추정"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["역명"] = df["역명"].astype(str).where(df["역명"].notna(), None)

    # 정렬
    df = df[required].sort_values(["기준_날짜", "시간", "정류장_순서", "정류장_ID"]).reset_index(drop=True)
    return df


def upsert_sqlite(db_path: Path, df: pd.DataFrame) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(CREATE_SQLITE)

    data = list(df.itertuples(index=False, name=None))
    cur.executemany(UPSERT_SQLITE, data)
    conn.commit()
    n = cur.rowcount if hasattr(cur, "rowcount") else len(data)
    conn.close()
    return n


def upsert_mysql(host: str, db: str, user: str, pwd: str, port: int, df: pd.DataFrame) -> int:
    if pymysql is None:
        raise RuntimeError("pymysql 이 설치되어 있지 않습니다. pip install pymysql")

    conn = pymysql.connect(host=host, user=user, password=pwd, database=db, port=port, charset="utf8mb4")
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_MYSQL)
            data = [
                (
                    r["기준_날짜"], int(r["시간"]), str(r["정류장_ID"]), int(r["정류장_순서"]),
                    (None if pd.isna(r["역명"]) else str(r["역명"])),
                    int(r["통과버스수"]) if pd.notna(r["통과버스수"]) else None,
                    float(r["예측_승차_총"]), float(r["예측_하차_총"]),
                    float(r["버스당_승차"]), float(r["버스당_하차"]), float(r["버스당_onboard_추정"])
                )
                for _, r in df.iterrows()
            ]
            cur.executemany(UPSERT_MYSQL, data)
        conn.commit()
    finally:
        conn.close()
    return len(df)


def s3_upload(local_path: Path, bucket: str, key: str):
    if boto3 is None:
        raise RuntimeError("boto3 가 설치되어 있지 않습니다. pip install boto3")
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", required=True, help="LSTM 예측 CSV 경로")
    # SQLite
    ap.add_argument("--sqlite", action="store_true", help="SQLite에 업서트 수행")
    ap.add_argument("--db-path", default="db/seoul_bus_172.db", help="SQLite DB 경로")
    # MySQL
    ap.add_argument("--mysql", action="store_true", help="MySQL(RDS)에 업서트 수행")
    ap.add_argument("--mysql-host", default=os.environ.get("DB_HOST"))
    ap.add_argument("--mysql-db", default=os.environ.get("DB_NAME"))
    ap.add_argument("--mysql-user", default=os.environ.get("DB_USER"))
    ap.add_argument("--mysql-pass", default=os.environ.get("DB_PASS"))
    ap.add_argument("--mysql-port", type=int, default=int(os.environ.get("DB_PORT", "3306")))
    # S3 업로드
    ap.add_argument("--s3-bucket", default=os.environ.get("S3_BUCKET"))
    ap.add_argument("--s3-prefix", default=os.environ.get("S3_PREFIX", "predictions"))
    ap.add_argument("--s3-upload-csv", action="store_true", help="CSV 원본 업로드")
    ap.add_argument("--s3-upload-db", action="store_true", help="SQLite DB 파일 업로드")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    # CSV 로드 & 정제
    df_raw = pd.read_csv(csv_path)
    df = coerce_and_align_columns(df_raw)

    total_upserted = 0

    # SQLite upsert
    if args.sqlite:
        n = upsert_sqlite(Path(args.db_path), df)
        total_upserted += n
        print(f"SQLite upsert 완료: {n} rows → {args.db_path}")

    # MySQL upsert
    if args.mysql:
        miss = [k for k in ["mysql_host", "mysql_db", "mysql_user", "mysql_pass"] if getattr(args, k) in (None, "")]
        if miss:
            raise ValueError(f"MySQL 접속 정보가 부족합니다: {miss}")
        n = upsert_mysql(args.mysql_host, args.mysql_db, args.mysql_user, args.mysql_pass, args.mysql_port, df)
        total_upserted += n
        print(f"MySQL upsert 완료: {n} rows → {args.mysql_host}/{args.mysql_db}.{TABLE_NAME}")

    if args.s3_bucket and args.s3_upload_csv:
        key = f"{args.s3_prefix.rstrip('/')}/csv/{csv_path.name}"
        s3_upload(csv_path, args.s3_bucket, key)
        print(f"☁️  S3 업로드(CSV): s3://{args.s3_bucket}/{key}")

    if args.s3_bucket and args.s3_upload_db and args.sqlite:
        dbp = Path(args.db_path)
        if dbp.exists():
            key = f"{args.s3_prefix.rstrip('/')}/sqlite/{dbp.name}"
            s3_upload(dbp, args.s3_bucket, key)
            print(f"☁️  S3 업로드(DB): s3://{args.s3_bucket}/{key}")
        else:
            print("⚠️ SQLite DB 파일이 없어 S3 업로드를 건너뜁니다.")

    print(f"총 upsert: {total_upserted} rows")
    print("완료.")

if __name__ == "__main__":
    main()