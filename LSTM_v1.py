# save as: /Users/minwoo/Desktop/softeer/data_engineering_course_materials/missions/final_project/LSTM_v1.py
# 목적: 정류장×시간 승차 수요 예측 (공급 피처 주입 버전)

import os
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------
# Paths
# --------------------------
DATA_CSV = "/Users/minwoo/Desktop/softeer/data_engineering_course_materials/missions/final_project/data/172_시간대별_정류장_노선_추정승하차+운행횟수+순서+링크거리.csv"
OUT_PRED = "/Users/minwoo/Desktop/softeer/data_engineering_course_materials/missions/final_project/data/172_LSTM_preds_정류장별_시간별_승하차_예측_혼잡도+추가배차.csv"

# --------------------------
# Train/Test 기간
# --------------------------
TRAIN_START = "2025-06-01"
TRAIN_END   = "2025-06-19"  # 학습 구간의 마지막 이틀은 검증으로 사용
VAL_START   = "2025-06-20"
VAL_END     = "2025-06-21"
TEST_START  = "2025-06-22"
TEST_END    = "2025-06-29"

SEQ_LEN = 24   # 24시간 히스토리 입력
BATCH   = 256
EPOCHS  = 100
LR      = 1e-3

np.random.seed(42)
tf.random.set_seed(42)

# --------------------------
# 1) 데이터 로드/정리
# --------------------------
df = pd.read_csv(DATA_CSV)

need = ["기준_날짜","시간","정류장_ID","정류장_순서","역명","승차인원","하차인원","운행횟수"]
miss = [c for c in need if c not in df.columns]
if miss:
    raise ValueError(f"입력에 필요한 컬럼 없음: {miss}")

# timestamp 생성
df["기준_날짜"] = pd.to_datetime(df["기준_날짜"])
df["timestamp"] = df["기준_날짜"] + pd.to_timedelta(df["시간"].astype(int), unit="h")

# 정렬
df = df.sort_values(["정류장_ID","timestamp"]).reset_index(drop=True)

# 통과버스수 = 시간대별 운행횟수
df["통과버스수"] = df["운행횟수"].fillna(0).clip(lower=0)

# 음수 승차/하차 보정(안전 클립)
df["승차인원"] = df["승차인원"].fillna(0).clip(lower=0)
df["하차인원"] = df["하차인원"].fillna(0).clip(lower=0)

# 링크거리 피처
if "링크_구간거리(m)" in df.columns:
    # 0은 실질적 결측으로 간주 → 평균 대체
    mean_nonzero = df.loc[df["링크_구간거리(m)"].replace(0, np.nan).notna(),"링크_구간거리(m)"] \
                     .replace(0, np.nan).mean()
    df["링크거리"] = df["링크_구간거리(m)"].replace(0, np.nan).fillna(mean_nonzero)
else:
    df["링크거리"] = 300.0

# 시간/요일/주말
df["hour"] = df["timestamp"].dt.hour
df["dow"]  = df["timestamp"].dt.dayofweek
df["is_wend"] = (df["dow"]>=5).astype(int)

# === 공급 피처 추가 ===
# headway(분) = 60/통과버스수, 0분모/결측 가드 + 이상치 캡핑
df["headway_min"] = np.where(df["통과버스수"] > 0, 60.0 / df["통과버스수"], np.nan)
df["headway_min"] = df["headway_min"].fillna(60.0).clip(lower=2.0, upper=60.0)

# 목표 버스당 탑승(좌석 33, 1.5배 기준)
CAPACITY = 33.0
TARGET_MULT = 1.5
df["target_per_bus"] = CAPACITY * TARGET_MULT  # 49.5 ≈ 50

# 정류장 임베딩용 인덱스
stop_ids = df["정류장_ID"].astype(str).unique()
stop2idx = {sid:i for i, sid in enumerate(stop_ids)}
df["stop_idx"] = df["정류장_ID"].astype(str).map(stop2idx)

# 표적: y_board = 승차, y_alight = 하차 (시간·정류장 총량)
df["y_board"]  = df["승차인원"].astype(float)
df["y_alight"] = df["하차인원"].astype(float)

# Train/Val/Test 마스크 (종료일 23:59:59 포함)
_end_train = pd.to_datetime(TRAIN_END) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
_end_val   = pd.to_datetime(VAL_END)   + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
_end_test  = pd.to_datetime(TEST_END)  + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

train_mask = (
    (df["timestamp"] >= pd.to_datetime(TRAIN_START)) &
    (df["timestamp"] <= _end_train)
)

val_mask = (
    (df["timestamp"] >= pd.to_datetime(VAL_START)) &
    (df["timestamp"] <= _end_val)
)

test_mask = (
    (df["timestamp"] >= pd.to_datetime(TEST_START)) &
    (df["timestamp"] <= _end_test)
)

# --------------------------
# 2) 시퀀스 빌더 (정류장별로 24시간 히스토리 → t 시점 예측)
# --------------------------
# ✅ 공급 피처 포함
feat_cols = [
    "hour","dow","is_wend",
    "정류장_순서","링크거리",
    "통과버스수","headway_min","target_per_bus"
]

def make_sequences(frame):
    X_num = []
    X_stop = []
    y = []
    keys = []
    for sid, g in frame.groupby("정류장_ID"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        arr = g[feat_cols].to_numpy(dtype=float)
        sid_idx = g["stop_idx"].iloc[0]
        yy = g[["y_board","y_alight"]].to_numpy(dtype=float)

        for t in range(SEQ_LEN, len(g)):
            X_num.append(arr[t-SEQ_LEN:t])             # (T,F)
            X_stop.append([sid_idx]*SEQ_LEN)            # (T,)
            y.append(yy[t])                             # (2,)
            keys.append((sid, g["timestamp"].iloc[t]))

    return (
        np.array(X_num),                               # (N,T,F)
        np.array(X_stop, dtype=int),                   # (N,T)
        np.array(y, dtype=float),
        keys
    )

train_df = df[train_mask].copy()
val_df   = df[val_mask].copy()
test_df  = df[test_mask].copy()

Xnum_tr, Xstop_tr, y_tr, _      = make_sequences(train_df)
Xnum_val, Xstop_val, y_val, _   = make_sequences(val_df)
Xnum_te, Xstop_te, y_te, keys_te = make_sequences(test_df)

# 정규화(Train 기준 MinMax)
mins = Xnum_tr.reshape(-1, Xnum_tr.shape[-1]).min(axis=0)
maxs = Xnum_tr.reshape(-1, Xnum_tr.shape[-1]).max(axis=0)
rng  = np.where((maxs-mins)==0, 1.0, (maxs-mins))

def norm_fn(x):
    return (x - mins) / rng

Xnum_tr  = norm_fn(Xnum_tr)
Xnum_val = norm_fn(Xnum_val)
Xnum_te  = norm_fn(Xnum_te)

# --------------------------
# 3) 모델 (정류장 임베딩 + 수치피처 → LSTM)
# --------------------------
num_features = Xnum_tr.shape[-1]
n_stops = len(stop2idx)

inp_num  = layers.Input(shape=(SEQ_LEN, num_features))
inp_stop = layers.Input(shape=(SEQ_LEN,), dtype="int32")

emb = layers.Embedding(input_dim=n_stops, output_dim=8)(inp_stop)  # (B,T,8)
x = layers.Concatenate(axis=-1)([inp_num, emb])                     # (B,T,F+8)
x = layers.LSTM(64, return_sequences=False)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(2, activation="relu")(x)  # [승차, 하차] 동시 예측

model = keras.Model([inp_num, inp_stop], out)
model.compile(optimizer=keras.optimizers.Adam(LR), loss="mae", metrics=["mse"])
model.summary()

# --------------------------
# 4) 학습
# --------------------------
callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5, verbose=1),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
]
model.fit([Xnum_tr, Xstop_tr], y_tr,
          validation_data=([Xnum_val, Xstop_val], y_val),
          epochs=EPOCHS, batch_size=BATCH, verbose=2,
          callbacks=callbacks)

# --------------------------
# 5) 예측 → CSV 생성 (test 기간)
# --------------------------
y_hat = model.predict([Xnum_te, Xstop_te], batch_size=BATCH, verbose=0)  # (N,2)
y_hat_board  = y_hat[:,0]
y_hat_alight = y_hat[:,1]

# 키를 DF로
pred_df = pd.DataFrame(keys_te, columns=["정류장_ID","timestamp"])
pred_df["정류장_ID"] = pred_df["정류장_ID"].astype(str)
pred_df["기준_날짜"] = pred_df["timestamp"].dt.strftime("%Y-%m-%d")
pred_df["시간"] = pred_df["timestamp"].dt.hour

# 메타 붙이기(순서/역명/통과버스수) - 같은 시점 행과 조인
join_cols = ["정류장_ID","기준_날짜","시간"]
meta = test_df[join_cols + ["정류장_순서","역명","통과버스수"]].drop_duplicates()

# 🔧 키 타입 정규화: 정류장_ID/기준_날짜/시간의 dtype을 맞춘다
meta = meta.copy()
meta["정류장_ID"] = meta["정류장_ID"].astype(str)
meta["기준_날짜"] = pd.to_datetime(meta["기준_날짜"]).dt.strftime("%Y-%m-%d")
meta["시간"] = meta["시간"].astype(int)

pred_df["시간"] = pred_df["시간"].astype(int)

pred_df = pred_df.merge(meta, on=join_cols, how="left")

# 총 승차/하차 예측 (시간·정류장 총량)
pred_df["예측_승차_총"]  = np.clip(y_hat_board,  0, None)
pred_df["예측_하차_총"] = np.clip(y_hat_alight, 0, None)

# 버스당 승차/하차 (0 분모 보호)
den = pred_df["통과버스수"].replace(0, np.nan)
pred_df["버스당_승차"]  = (pred_df["예측_승차_총"]  / den).fillna(0.0)
pred_df["버스당_하차"] = (pred_df["예측_하차_총"] / den).fillna(0.0)

# 같은 (날짜,시간) 블록 내에서 정류장 순서대로 누적 → 버스당 onboard 추정
pred_df = pred_df.sort_values(["기준_날짜","시간","정류장_순서"]).reset_index(drop=True)

def _accumulate_onboard(g):
    onboard = (g["버스당_승차"] - g["버스당_하차"]).cumsum()
    onboard = onboard.clip(lower=0.0)
    g["버스당_onboard_추정"] = onboard
    return g

pred_df = pred_df.groupby(["기준_날짜","시간"], group_keys=False).apply(_accumulate_onboard)

# 최종 컬럼 & 정렬
out = pred_df[[
    "기준_날짜","시간","정류장_ID","정류장_순서","역명","통과버스수",
    "예측_승차_총","예측_하차_총","버스당_승차","버스당_하차","버스당_onboard_추정"
]].copy()
out = out.sort_values(["기준_날짜","시간","정류장_순서"]).reset_index(drop=True)

# 저장
Path(os.path.dirname(OUT_PRED)).mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_PRED, index=False, encoding="utf-8-sig")
print("✅ 예측 CSV 저장:", OUT_PRED)