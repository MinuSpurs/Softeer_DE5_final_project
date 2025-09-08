# 원천 TPSS CSV(넓은 형태: 버스운행횟수_00시~23시)를 S3에서 읽어
# S3 Parquet(Bronze)로 저장. 날짜/노선 필터도 즉석에서 가능.

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, regexp_replace, trim
import os

# 입력/출력 경로 (필요 시 인자로 바꿔도 됨)
S3_INPUT  = os.getenv("S3_TPSS_CSV",  "s3://seoul-bus-analytics/uploads/tpss/2025-06/tpss_sta_route_hturn_merged.csv")
S3_OUTPUT = os.getenv("S3_BRONZE",    "s3://seoul-bus-analytics/bronze/tpss")

# 필터
USE_YM_FILTER   = os.getenv("USE_YM", "202506")    
ROUTE_NO_FILTER = os.getenv("ROUTE_NO", "172")  

spark = (
    SparkSession.builder
    .appName("bronze_ingest_tpss")
    .getOrCreate()
)

df = (
    spark.read
    .option("header", "true")
    .option("encoding", "UTF-8")
    .csv(S3_INPUT)
)

# 컬럼 정규화: 노선번호/노선_ID/정류장_ID/정류장_순서/기준_날짜 등 네이밍 통일
# 아래 컬럼명은 네 파일 스키마에 맞게 조정
df = (
    df
    .withColumn("기준_날짜", trim(col("기준_날짜")))
    .withColumn("노선_ID", trim(col("노선_ID")))
    .withColumn("정류장_ID", trim(col("정류장_ID")))
    .withColumn("노선번호", trim(col("노선번호")))
    .withColumn("정류장_순서", col("정류장_순서").cast("int"))
)

# 월 필터(YYYYMM), 노선번호 필터
df = df.where(regexp_replace(col("기준_날짜"), "-", "").substr(1, 6) == USE_YM_FILTER)
df = df.where(col("노선번호") == lit(ROUTE_NO_FILTER))

# 파티셔닝 컬럼 생성
df = df.withColumn("use_ym", regexp_replace(col("기준_날짜"), "-", "").substr(1, 6))
df = df.withColumn("date",   regexp_replace(col("기준_날짜"), "-", ""))

# 저장 (Overwrite 파티션)
(
    df.write
      .mode("overwrite")
      .partitionBy("use_ym", "date")
      .parquet(S3_OUTPUT)
)

print("bronze saved:", S3_OUTPUT)
spark.stop()