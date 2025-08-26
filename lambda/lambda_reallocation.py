# lambda/lambda_reallocation.py
import os, json, logging
import pymysql

logging.getLogger().setLevel(logging.INFO)

DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DB_NAME")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))

def _conn():
    if not all([DB_HOST, DB_USER, DB_PASS, DB_NAME]):
        raise RuntimeError("DB env vars(DB_HOST/DB_USER/DB_PASS/DB_NAME) not set")
    return pymysql.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASS,
        database=DB_NAME, port=DB_PORT, autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10, read_timeout=10, write_timeout=10
    )

def handler(event, context):
    """
    기본: 헬스체크 + DB 연결 확인.
    event 예) {"action":"health"} 또는 {"action":"summary","route_no":"172","date":"2025-06-23"}
    """
    action = (event or {}).get("action", "health")

    try:
        with _conn() as conn, conn.cursor() as cur:
            # DB 시간
            cur.execute("SELECT NOW() AS now")
            now = cur.fetchone()["now"]

            if action == "health":
                return {"ok": True, "action": action, "db_time": str(now)}

            if action == "summary":
                route_no = (event or {}).get("route_no", "172")
                date_str = (event or {}).get("date")  # "YYYY-MM-DD"
                # 스키마에 따라 바꾸세요. 컬럼명이 한글이면 백틱(`)으로 감싸야 합니다.
                # 예: SELECT `시간`, SUM(`운행횟수`) ...
                sql = """
                    SELECT 시간 AS hour, SUM(운행횟수) AS departures
                    FROM daily_stop_hour
                    WHERE 노선번호=%s AND 기준_날짜=%s
                    GROUP BY 시간
                    ORDER BY 시간
                """
                cur.execute(sql, (route_no, date_str))
                hours = cur.fetchall()
                return {"ok": True, "action": action, "route_no": route_no, "date": date_str, "hours": hours, "db_time": str(now)}

            return {"ok": True, "action": action, "db_time": str(now)}

    except Exception as e:
        logging.exception("Lambda error")
        return {"ok": False, "error": str(e)}