# lambda/lambda_reallocation.py
# Purpose:
# - Health check and simple read APIs from MySQL (RDS).
# - Supports API Gateway HTTP API v2 (proxy) and direct Lambda invoke.
#
# Endpoints (intended):
#   GET /v1/routes/{route}/days/{date}/summary
#   GET /v1/routes/{route}/days/{date}/stops/{stop_seq}
#   GET /v1/routes/{route}/forecast?start=YYYY-MM-DD&days=N
#
# Query-string fallback also supported:
#   /summary?route_no=172&date=2025-06-23
#   /stops?route_no=172&date=2025-06-23&stop_seq=1
#   /forecast?route_no=172&start=2025-06-24&days=7
#
# Env Vars required:
#   DB_HOST, DB_USER, DB_PASS, DB_NAME [, DB_PORT]
#
import os
import json
import logging
import pymysql
from decimal import Decimal
from datetime import date, datetime
import boto3
import tempfile
import subprocess
import datetime as _dt

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DB_NAME")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "predictions")

# --------------------------------
# DB Connection
# --------------------------------
def _conn():
    if not all([DB_HOST, DB_USER, DB_PASS, DB_NAME]):
        raise RuntimeError("DB env vars(DB_HOST/DB_USER/DB_PASS/DB_NAME) not set")
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        port=DB_PORT,
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10,
        charset="utf8mb4",
    )

# --------------------------------
# JSON-safe converter
# --------------------------------
def _json_safe(obj):
    """Recursively convert types (Decimal, datetime, bytes) into JSON-friendly values."""
    if isinstance(obj, Decimal):
        # convert DECIMAL to float (or use str(obj) if you want exact string)
        return float(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.decode("latin1", errors="ignore")
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj

# --------------------------------
# Helpers for API Gateway v2 response
# --------------------------------
def _resp(status_code: int, body: dict):
    """HTTP-style response (API Gateway v2 compatible)."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json; charset=utf-8",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(_json_safe(body), ensure_ascii=False),
    }

def _ok(body: dict):
    return _resp(200, body)

def _bad_request(msg: str):
    return _resp(400, {"ok": False, "error": msg})

def _server_error(msg: str):
    return _resp(500, {"ok": False, "error": msg})

def _response(body: dict, event: dict):
    """Return API Gateway style or direct response for non-HTTP invokes."""
    if isinstance(event, dict) and "version" in event:
        return _ok(body)
    return _json_safe(body)

def _download_s3(bucket: str, key: str, local_path: str):
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)

def _run_insert_script(local_csv: str):
    # Runs the packaged ETL script to upsert CSV into MySQL (RDS)
    # The script should read DB_* env vars itself or accept --mysql flag as implemented previously.
    cmd = ["python", "etl/insert_lstm_predictions.py", "--csv-path", local_csv, "--mysql"]
    subprocess.check_call(cmd)

# --------------------------------
# Param parsing (API GW v2 or direct)
# --------------------------------
def _get_params(event: dict):
    """
    Returns: dict with:
      action, route_no, date, stop_seq, start, days, rte_id
    Supports:
      - API GW v2: queryStringParameters + pathParameters + rawPath
      - direct invoke: top-level keys
    """
    out = {
        "action": "health",
        "route_no": None,
        "date": None,
        "stop_seq": None,
        "start": None,
        "days": None,
        "rte_id": None,
        "s3_bucket": None,
        "s3_key": None,
        "s3_prefix": None,
        "date_offset_days": None,
    }

    if not isinstance(event, dict):
        return out

    qs = event.get("queryStringParameters") or {}
    path_params = event.get("pathParameters") or {}
    raw_path = event.get("rawPath") or ""

    # action inference from path or explicit
    action = event.get("action") or qs.get("action")
    if not action:
        if raw_path.endswith("/summary"):
            action = "summary"
        elif "/stops/" in raw_path or raw_path.endswith("/stops"):
            action = "stop_detail"
        elif raw_path.endswith("/forecast"):
            action = "forecast"
        else:
            action = "health"
    out["action"] = action

    # route/date/stop_seq (path or query)
    out["route_no"] = (
        event.get("route_no")
        or path_params.get("route")
        or qs.get("route_no")
        or qs.get("routeNo")
        or "172"
    )
    out["date"] = (
        event.get("date")
        or path_params.get("date")
        or qs.get("date")
    )
    out["stop_seq"] = (
        event.get("stop_seq")
        or path_params.get("stop_seq")
        or path_params.get("stopSeq")
        or qs.get("stop_seq")
        or qs.get("stopSeq")
    )
    out["rte_id"] = event.get("rte_id") or qs.get("rte_id")

    # forecast params
    out["start"] = event.get("start") or qs.get("start")
    try:
        out["days"] = int(event.get("days") or qs.get("days") or 7)
    except Exception:
        out["days"] = 7

    out["s3_bucket"] = event.get("s3_bucket") or qs.get("s3_bucket") or S3_BUCKET
    out["s3_key"] = event.get("s3_key") or qs.get("s3_key")
    out["s3_prefix"] = event.get("s3_prefix") or qs.get("s3_prefix") or S3_PREFIX
    try:
        out["date_offset_days"] = int(event.get("date_offset_days") or qs.get("date_offset_days") or 1)
    except Exception:
        out["date_offset_days"] = 1

    return out

# --------------------------------
# SQL helpers
# --------------------------------
SQL_SUMMARY = """
SELECT
  `시간` AS hour,
  SUM(`운행횟수`) AS current_departures,
  CASE WHEN SUM(`운행횟수`) > 0 THEN 60.0 / SUM(`운행횟수`) END AS headway_min,
  MAX(CASE WHEN `운행횟수` > 0 THEN `승차인원` / `운행횟수` END) AS max_board_per_bus,
  CASE
    WHEN MAX(CASE WHEN `운행횟수` > 0 THEN `승차인원` / `운행횟수` END) > 49.5
    THEN 1 ELSE 0
  END AS is_crowded
FROM daily_stop_hour
WHERE `노선번호`=%s AND `기준_날짜`=%s
GROUP BY `시간`
ORDER BY `시간`;
"""

SQL_STOP_DETAIL = """
SELECT
  `시간` AS hour,
  SUM(`운행횟수`) AS pass_buses,
  SUM(`승차인원`) AS board_total,
  SUM(`하차인원`) AS alight_total,
  CASE WHEN SUM(`운행횟수`) > 0 THEN SUM(`승차인원`) / SUM(`운행횟수`) END AS board_per_bus,
  MAX(`링크_구간거리_m`) AS link_distance_m,
  COALESCE(MIN(`역명`), NULL) AS stop_name
FROM daily_stop_hour
WHERE `노선번호`=%s AND `기준_날짜`=%s AND `정류장_순서`=%s
GROUP BY `시간`
ORDER BY `시간`;
"""


# --------------------------------
# Handler
# --------------------------------
def handler(event, context):
    """
    API patterns:
      - GET /health
      - GET /v1/routes/{route}/days/{date}/summary
      - GET /v1/routes/{route}/days/{date}/stops/{stop_seq}
      - GET /v1/routes/{route}/forecast?start=YYYY-MM-DD&days=N
    """
    try:
        params = _get_params(event or {})
        logger.info({"event": event, "parsed": params})

        with _conn() as conn, conn.cursor() as cur:
            # DB time check (also used in payload)
            cur.execute("SELECT NOW() AS now")
            now = str(cur.fetchone()["now"])

            action = params["action"]

            # Health
            if action == "health":
                if isinstance(event, dict) and "version" in event:
                    return _ok({"ok": True, "action": "health", "db_time": now})
                return {"ok": True, "action": "health", "db_time": now}

            # Summary
            if action == "summary":
                route_no = params["route_no"]
                date_str = params["date"]
                if not date_str:
                    msg = "missing 'date' (YYYY-MM-DD)"
                    return _bad_request(msg) if "version" in (event or {}) else {"ok": False, "error": msg}

                cur.execute(SQL_SUMMARY, (route_no, date_str))
                rows = cur.fetchall() or []
                payload = {
                    "ok": True,
                    "action": "summary",
                    "route_no": route_no,
                    "date": date_str,
                    "hours": rows,
                    "db_time": now,
                }
                return _ok(payload) if "version" in (event or {}) else _json_safe(payload)

            # Stop detail
            if action in ("stop_detail", "stops"):
                route_no = params["route_no"]
                date_str = params["date"]
                stop_seq = params["stop_seq"]
                if not date_str or stop_seq is None:
                    msg = "missing params: require 'date' (YYYY-MM-DD) and 'stop_seq'"
                    return _bad_request(msg) if "version" in (event or {}) else {"ok": False, "error": msg}

                cur.execute(SQL_STOP_DETAIL, (route_no, date_str, int(stop_seq)))
                rows = cur.fetchall() or []
                payload = {
                    "ok": True,
                    "action": "stop_detail",
                    "route_no": route_no,
                    "date": date_str,
                    "stop_seq": int(stop_seq),
                    "hours": rows,
                    "db_time": now,
                }
                return _ok(payload) if "version" in (event or {}) else _json_safe(payload)

            # Forecast (weekday average fallback)
            # Forecast (weekday-average; Python-side date expansion)
            if action == "forecast":
                route_no = params["route_no"]
                start = params["start"]
                days = params["days"] or 7
                if not start:
                    msg = "missing 'start' (YYYY-MM-DD)"
                    return _bad_request(msg) if "version" in (event or {}) else {"ok": False, "error": msg}

                # 1) 요일 × 시간대 평균  (DB에서 한 번만)
                cur.execute("""
                    SELECT
                      DAYOFWEEK(`기준_날짜`) AS dow,
                      `시간` AS hour,
                      AVG(`승차인원`) AS avg_board,
                      AVG(`운행횟수`) AS avg_dep,
                      AVG(CASE WHEN `운행횟수`>0 THEN `승차인원`/`운행횟수` END) AS avg_board_per_bus
                    FROM daily_stop_hour
                    WHERE `노선번호`=%s
                    GROUP BY DAYOFWEEK(`기준_날짜`), `시간`
                """, (route_no,))
                avg_rows = cur.fetchall() or []

                # 2) 파이썬에서 날짜 목록 만들고 요일별 평균 매핑
                import datetime as _dt

                start_dt = _dt.datetime.strptime(start, "%Y-%m-%d").date()
                date_list = [start_dt + _dt.timedelta(days=i) for i in range(int(days))]

                # dow -> hour -> stats 맵
                # MySQL DAYOFWEEK: 1=일, 2=월, ... 7=토
                dow_map = {}
                for r in avg_rows:
                    d = int(r["dow"])
                    h = int(r["hour"])
                    dow_map.setdefault(d, {})[h] = {
                        "avg_dep": r["avg_dep"] or 0.0,
                        "avg_board_per_bus": r["avg_board_per_bus"] or 0.0,
                    }

                out_rows = []
                for d in date_list:
                    # python: Monday=0..Sunday=6 -> MySQL: Sunday=1..Saturday=7
                    py_w = d.weekday()            # Mon=0..Sun=6
                    mysql_dow = 1 if py_w == 6 else (py_w + 2)
                    per_hour = dow_map.get(mysql_dow, {})

                    for hour in range(24):
                        stats = per_hour.get(hour, {"avg_dep": 0.0, "avg_board_per_bus": 0.0})
                        dep = float(stats["avg_dep"] or 0.0)
                        bpb = float(stats["avg_board_per_bus"] or 0.0)
                        headway = (60.0/dep) if dep > 0 else None
                        crowded = 1 if bpb > 49.5 else 0
                        out_rows.append({
                            "route_no": route_no,
                            "date": d.strftime("%Y-%m-%d"),
                            "hour": hour,
                            "forecast_departures": dep,
                            "forecast_headway_min": headway,
                            "forecast_board_per_bus": bpb,
                            "crowded_flag": crowded,
                        })

                payload = {
                    "ok": True,
                    "action": "forecast",
                    "route_no": route_no,
                    "start": start,
                    "days": int(days),
                    "hours": out_rows,
                    "db_time": now,
                }
                return _ok(payload) if "version" in (event or {}) else _json_safe(payload)

            # Ingest predictions from S3 and upsert into RDS
            if action in ("ingest", "ingest_preds", "ingest_predictions"):
                # file selection priority: explicit s3_key > (prefix + date pattern)
                s3_bucket = params.get("s3_bucket") or S3_BUCKET
                s3_key = params.get("s3_key")
                s3_prefix = params.get("s3_prefix") or S3_PREFIX

                if not s3_key:
                    # build default key using date (or date_offset_days)
                    if not params.get("date"):
                        # default: use (UTC-1day) for yesterday
                        d = (_dt.datetime.utcnow().date() - _dt.timedelta(days=int(params.get("date_offset_days") or 1)))
                        date_str = d.strftime("%Y-%m-%d")
                    else:
                        date_str = params["date"]
                    # NOTE: adjust filename pattern to your actual one if different
                    filename = f"172_LSTM_preds_{date_str}.csv"
                    s3_key = f"{s3_prefix.rstrip('/')}/{filename}"

                if not s3_bucket or not s3_key:
                    msg = "missing S3 info (s3_bucket and s3_key or s3_prefix+date)"
                    return _bad_request(msg) if "version" in (event or {}) else {"ok": False, "error": msg}

                # download to /tmp and run upsert
                with tempfile.TemporaryDirectory() as td:
                    local_csv = os.path.join(td, os.path.basename(s3_key))
                    try:
                        _download_s3(s3_bucket, s3_key, local_csv)
                    except Exception as e:
                        msg = f"failed to download s3://{s3_bucket}/{s3_key}: {e}"
                        return _server_error(msg) if "version" in (event or {}) else {"ok": False, "error": msg}

                    try:
                        _run_insert_script(local_csv)
                    except subprocess.CalledProcessError as e:
                        msg = f"upsert script failed: {e}"
                        return _server_error(msg) if "version" in (event or {}) else {"ok": False, "error": msg}

                payload = {
                    "ok": True,
                    "action": "ingest",
                    "s3_bucket": s3_bucket,
                    "s3_key": s3_key,
                    "db_time": now,
                }
                return _ok(payload) if "version" in (event or {}) else _json_safe(payload)

            # default fallback
            if "version" in (event or {}):
                return _ok({"ok": True, "action": action, "db_time": now})
            return {"ok": True, "action": action, "db_time": now}

    except Exception as e:
        logger.exception("Lambda error")
        if isinstance(event, dict) and "version" in event:
            return _server_error(str(e))
        return {"ok": False, "error": str(e)}

# Usage examples (API Gateway or direct invoke):
#   POST/GET /v1/ingest?s3_bucket=seoul-bus-analytics&s3_key=predictions/172_LSTM_preds_2025-06-23.csv
#   POST body: {"action":"ingest","date":"2025-06-23"}  # builds key from S3_PREFIX + pattern