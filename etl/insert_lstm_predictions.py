# save as: etl/insert_lstm_predictions.py
import argparse
import sqlite3
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", required=True, help="SQLite DB path")
    ap.add_argument("--csv-path", required=True, help="CSV predictions path")
    args = ap.parse_args()

    # CSV 로드
    df = pd.read_csv(args.csv_path)

    conn = sqlite3.connect(args.db_path)
    cur = conn.cursor()

    # 테이블 생성
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fact_lstm_predictions (
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
    )
    """)

    insert_sql = """
    INSERT INTO fact_lstm_predictions (
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
    """

    data_to_insert = df.itertuples(index=False, name=None)
    cur.executemany(insert_sql, data_to_insert)

    conn.commit()
    conn.close()
    print(f"✅ LSTM predictions inserted into {args.db_path}")

if __name__ == "__main__":
    main()