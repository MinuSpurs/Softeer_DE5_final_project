# -*- coding: utf-8 -*-
# Silver의 ops_hourly를 읽어 API가 필요로 하는 요약을 MySQL에 적재.
# 테이블:
#   api_hour_summary(route_no, date, hour, current_departures)
#   api_stop_hour_detail(route_no, date, stop_id, stop_seq, hour, departures)
#
# ※ 처음 한 번 수동으로 스키마 생성(EC2에서 mysql 접속 후):
# CREATE DATABASE seoul_bus_172;
# USE seoul_bus_172;
# CREATE TABLE IF NOT EXISTS api_hour_summary(
#   route_no VARCHAR(16), date CHAR(8), hour INT, current_departures DOUBLE,
#   PRIMARY KEY(route_no, date, hour)
# );
# CREATE TABLE IF NOT EXISTS api_stop_hour_detail(
#   route_no VARCHAR(16), date CHAR(8), stop_id VARCHAR(32), stop_seq INT, hour INT, departures DOUBLE,
#   PRIMARY KEY(route_no, date, stop_id, hour)
# );

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum
import os

S3_SILVER = os.getenv("S3_SILVER",  "s3://seoul-bus-analytics/silver/ops_hourly")
USE_YM    = os.getenv("USE_YM",     "202506")
ROUTE_NO  = os.getenv("ROUTE_NO",   "172")

# JDBC (환경변수로 관리)
JDBC_URL  = os.getenv("JDBC_URL")   # e.g. jdbc:mysql://bus-172.xxxx.ap-northeast-2.rds.amazonaws.com:3306/seoul_bus_172?useSSL=false
DB_USER   = os.getenv("DB_USER")    # admin
DB_PASS   = os.getenv("DB_PASS")    # ****
DB_DRIVER = "com.mysql.cj.jdbc.Driver"

spark = (
    SparkSession.builder
    .appName("gold_publish_mysql")
    .config("spark.jars.packages", "mysql:mysql-connector-java:8.0.33")
    .getOrCreate()
)

silver = (
    spark.read.parquet(S3_SILVER)
    .where(col("use_ym") == USE_YM)
    .where(col("노선번호") == ROUTE_NO)
)

# --- 3-1) 시간 요약(API summary용) ---
hour_summary = (
    silver.groupBy("노선번호","date","시간")
          .agg(_sum("운행횟수").alias("current_departures"))
          .withColumnRenamed("노선번호","route_no")
          .withColumnRenamed("시간","hour")
)

# --- 3-2) 정류장 상세(API stop 상세용) ---
stop_detail = (
    silver.groupBy("노선번호","date","정류장_ID","정류장_순서","시간")
          .agg(_sum("운행횟수").alias("departures"))
          .withColumnRenamed("노선번호","route_no")
          .withColumnRenamed("정류장_ID","stop_id")
          .withColumnRenamed("정류장_순서","stop_seq")
          .withColumnRenamed("시간","hour")
)

# --- JDBC 쓰기 (overwrite → 필요 시 upsert 전략으로 변경 가능) ---
# 시간 요약
(
    hour_summary.write
      .format("jdbc")
      .option("url", JDBC_URL)
      .option("dbtable", "api_hour_summary")
      .option("user", DB_USER)
      .option("password", DB_PASS)
      .option("driver", DB_DRIVER)
      .mode("append")   # 초기에 empty면 overwrite도 OK. 운영은 append + PK 충돌시 REPLACE 쿼리 사용 권장
      .save()
)

# 정류장 상세
(
    stop_detail.write
      .format("jdbc")
      .option("url", JDBC_URL)
      .option("dbtable", "api_stop_hour_detail")
      .option("user", DB_USER)
      .option("password", DB_PASS)
      .option("driver", DB_DRIVER)
      .mode("append")
      .save()
)

print("✅ gold published to MySQL")
spark.stop()