# -*- coding: utf-8 -*-
# Silver의 ops_hourly를 읽어 API가 필요로 하는 요약을 MySQL에 적재.

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum
import os

S3_SILVER = os.getenv("S3_SILVER",  "s3://seoul-bus-analytics/silver/ops_hourly")
USE_YM    = os.getenv("USE_YM",     "202506")
ROUTE_NO  = os.getenv("ROUTE_NO",   "172")

# JDBC (환경변수로 관리)
JDBC_URL  = os.getenv("JDBC_URL") 
DB_USER   = os.getenv("DB_USER")   
DB_PASS   = os.getenv("DB_PASS")    
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

print("gold published to MySQL")
spark.stop()