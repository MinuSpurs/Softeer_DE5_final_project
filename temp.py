import pandas as pd
from pathlib import Path

# 원본 파일 경로
src = Path("/Users/minwoo/Desktop/softeer/data_engineering_course_materials/missions/final_project/result/172_정시성개선_권장출발대수_하루단위_탐욕안.csv")

# 1) 로드 및 타입 정리
df = pd.read_csv(src)
for c in ["권장_출발대수","기존_출발대수","증감"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

# 2) 시간별 추가배차(음수는 0으로 클립): "그 시간에 실제로 더 필요한 대수"
df["시간별_추가배차"] = (df["권장_출발대수"] - df["기존_출발대수"]).clip(lower=0)

# 3) 일일 합계(날짜별 스칼라). 이게 그날 전체로 아직 필요한 최소 추가대수의 합(시간 기준)입니다.
daily_sum = (
    df.groupby("기준_날짜", as_index=False)["시간별_추가배차"]
      .sum()
      .rename(columns={"시간별_추가배차":"일일_추가배차_합"})
)

# 4) 조인: 날짜별 스칼라를 시간행에 붙임(복제 표시)
out = df.merge(daily_sum, on="기준_날짜", how="left")

# 5) (선택) 혼동되는 기존 컬럼 제거/교체
if "총_추가배차_스코어(개선후)" in out.columns:
    out = out.drop(columns=["총_추가배차_스코어(개선후)"])

# 6) 파일 분리 저장: 시간별 파일 / 일일요약 파일
dst_hourly = src.with_name(src.stem + "_시간별정리.csv")
dst_daily  = src.with_name(src.stem + "_일일요약.csv")

# 시간별: 시간 단위 의사결정에 필요한 모든 컬럼 + 일일 합(복제) 포함
out.to_csv(dst_hourly, index=False, encoding="utf-8-sig")

# 일일요약: 날짜-별로 하나의 행만
daily_sum.to_csv(dst_daily, index=False, encoding="utf-8-sig")

print(f"✅ 저장(시간별): {dst_hourly}")
print(f"✅ 저장(일일요약): {dst_daily}")

# 7) 검증: 2025-06-26 하루 체크
day = "2025-06-26"
chk = out[out["기준_날짜"]==day].copy()

print("\n[검증] 2025-06-26 시간별 증감/추가배차 요약")
print(chk[["시간","권장_출발대수","기존_출발대수","증감","시간별_추가배차"]])

print("\n[검증] 2025-06-26 합계")
print("시간별_추가배차 합:", chk["시간별_추가배차"].sum())
print("증감 합(참고):", chk["증감"].sum())  # 재배치라면 0 또는 0에 근접
print("일일_추가배차_합(복제):", chk["일일_추가배차_합"].iloc[0])