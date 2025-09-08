# Bronze Parquet에서 wide → long 변환


from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, regexp_replace, col, lit, sequence, to_timestamp, explode, array
import os

S3_BRONZE = os.getenv("S3_BRONZE", "s3://seoul-bus-analytics/bronze/tpss")
S3_SILVER = os.getenv("S3_SILVER", "s3://seoul-bus-analytics/silver/ops_hourly")

USE_YM     = os.getenv("USE_YM", "202506")
ROUTE_NO   = os.getenv("ROUTE_NO", "172")

spark = (
    SparkSession.builder
    .appName("silver_ops_hourly_from_tpss")
    .getOrCreate()
)

bronze = spark.read.parquet(S3_BRONZE).where(col("use_ym") == USE_YM).where(col("노선번호") == ROUTE_NO)

# 시간대 컬럼 이름들
hour_cols = [f"버스운행횟수_{h:02d}시" for h in range(24) if f"버스운행횟수_{h:02d}시" in bronze.columns]

pairs = array(*[
    expr(f"named_struct('hour',{h}, '운행횟수', cast(`버스운행횟수_{h:02d}시` as double))")
    for h in range(24) if f"버스운행횟수_{h:02d}시" in bronze.columns
])

long_df = (
    bronze
    .withColumn("pairs", pairs)
    .select("기준_날짜","노선_ID","정류장_ID","정류장_순서","노선번호","use_ym","date","pairs")
    .withColumn("kv", explode(col("pairs")))
    .select(
        "기준_날짜","노선_ID","정류장_ID","정류장_순서","노선번호","use_ym","date",
        col("kv.hour").alias("시간"),
        col("kv.운행횟수").alias("운행횟수")
    )
)

(
    long_df.write
      .mode("overwrite")
      .partitionBy("use_ym","노선번호","date")
      .parquet(S3_SILVER)
)

print("silver saved:", S3_SILVER)
spark.stop()