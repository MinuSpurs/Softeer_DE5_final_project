import numpy as np
import pandas as pd
from math import ceil

# ------------------------
# 설정
# ------------------------
PRED_CSV = "/Users/minwoo/Desktop/softeer/data_engineering_course_materials/missions/final_project/data/172_LSTM_preds_정류장별_시간별_승하차_예측.csv"
INPUT_CSV = "/Users/minwoo/Desktop/softeer/data_engineering_course_materials/missions/final_project/data/172_시간대별_정류장_노선_추정승하차+운행횟수+순서+링크거리_보정.csv"

SEAT = 33
CROWD_MULT = 1.5
CAP = SEAT * CROWD_MULT       # 버스당 허용 목표(혼잡 회피 상한)
BASE_DWELL_MIN = 0.3          # 정차시간(분)
MIN_COVERAGE = 1              # 시간당 최소 출발대수(원하면 0도 가능)
SHIFT_LIMIT_PER_HOUR = 2      # 한 시간에서 늘이거나 줄일 수 있는 최대 대수 (현실 제약용)

# 속도 프로파일 (시간대별 평균속도; 필요시 조정)
def speed_m_per_min(hour):
    kmh = 15 if 7 <= hour <= 9 else 22
    return kmh * 1000.0 / 60.0

# ------------------------
# 데이터 로드
# ------------------------
pred_df = pd.read_csv(PRED_CSV, parse_dates=['기준_날짜'])
pred_df['시간'] = pred_df['시간'].astype(int)

raw = pd.read_csv(INPUT_CSV, parse_dates=['기준_날짜'])
raw['시간'] = raw['시간'].astype(int)

# 정류장 순서/링크 거리 마스터
stops = (raw[['정류장_ID','정류장_순서','역명','링크_구간거리(m)']]
         .drop_duplicates(subset=['정류장_ID','정류장_순서'])
         .sort_values('정류장_순서'))
stop_ids = stops['정류장_ID'].tolist()
S = int(stops['정류장_순서'].max())

first_stop_order = int(stops['정류장_순서'].min())

# 실제(현행) 시간별 첫 정류장 출발 대수(=운행횟수) 추출
panel = raw.copy()
panel['timestamp'] = panel['기준_날짜'] + pd.to_timedelta(panel['시간'], unit='h')
# 첫 정류장만
first_stop_id = stops.loc[stops['정류장_순서']==first_stop_order, '정류장_ID'].iloc[0]
now_departures = (panel[panel['정류장_ID'] == first_stop_id]
                  .groupby(['기준_날짜','시간'])['운행횟수']
                  .sum().astype(int).reset_index())
# 전체 날짜 목록
days = sorted(now_departures['기준_날짜'].dt.date.unique())

# 혼잡 평가에 쓰일 예측 총량 (정류장×시간 합산)
pred_totals = pred_df.groupby(['기준_날짜','시간','정류장_순서'], as_index=False)[['예측_승차인원','예측_하차인원']].sum()

# 링크 거리 look-up
link_by_order = stops.set_index('정류장_순서')['링크_구간거리(m)'].to_dict()

def simulate_passes(day, dep_map):
    """
    dep_map: dict[(date,hour)] = 첫 정류장 출발 대수(int)
    반환: passes_aug (버스 통과 이벤트 + 균등 분배 board_i/alight_i + bus_onboard)
         M_df(통과버스수), avg_onboard(버스당 평균 탑승/혼잡도/증차 필요)
    """
    # 시뮬에 들어갈 시간 집합: 해당 날짜의 모든 시간
    hours = list(range(0,24))
    base_ts = pd.Timestamp(day)

    # 통과 이벤트 생성
    passes = []
    for h in hours:
        N = int(dep_map.get((day, h), 0))
        if N <= 0:
            continue
        headway = 60.0 / N
        for k in range(N):
            abs_min = (pd.Timestamp(day) + pd.to_timedelta(h, 'h') - base_ts).total_seconds()/60.0 + k*headway
            cur_min = abs_min
            cur_order = first_stop_order
            bus_id = f"{day}-{h:02d}-{k+1}"

            # 첫 정류장 이벤트
            passes.append({'abs_min':cur_min,'기준_날짜':day,'시간':h,'정류장_순서':cur_order,'bus_id':bus_id})

            # 진행
            for s in range(first_stop_order, S):
                hr = int(cur_min // 60) % 24
                v = max(speed_m_per_min(hr), 1e-6)
                link_m = link_by_order.get(s+1, np.nan)
                if pd.isna(link_m):
                    link_m = np.nanmean([v for v in link_by_order.values() if pd.notna(v)])  # 평균 대치
                run_min = float(link_m) / v
                cur_min = cur_min + run_min + BASE_DWELL_MIN

                # 도착 이벤트
                t_abs = base_ts + pd.to_timedelta(cur_min, unit='m')
                passes.append({
                    'abs_min':cur_min,
                    '기준_날짜': t_abs.date(),
                    '시간': t_abs.hour,
                    '정류장_순서': s+1,
                    'bus_id': bus_id
                })

    if not passes:
        # 해당 날짜 운행 없음
        cols = ['abs_min','기준_날짜','시간','정류장_순서','bus_id','통과버스수','예측_승차인원','예측_하차인원','board_i','alight_i','버스별_onboard']
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=['기준_날짜','시간','정류장_순서','통과버스수']), pd.DataFrame(), pd.DataFrame()

    passes_df = pd.DataFrame(passes)
    # 날짜 dtype 통일 (object/date -> datetime64[ns])
    passes_df['기준_날짜'] = pd.to_datetime(passes_df['기준_날짜'])

    # 해당 날짜의 예측 붙이기
    pred_d = pred_totals[pred_totals['기준_날짜'].dt.date == day].copy()
    # 안전장치: (날짜·시간·순서) 단위 중복 제거(합산)
    pred_d = (pred_d.groupby(['기준_날짜','시간','정류장_순서'], as_index=False)
                    [['예측_승차인원','예측_하차인원']].sum())
    M_df = (passes_df.groupby(['기준_날짜','시간','정류장_순서'])['bus_id']
                    .nunique()
                    .reset_index(name='통과버스수'))
    # merge 키들의 dtype을 모두 datetime64[ns]로 통일
    M_df['기준_날짜'] = pd.to_datetime(M_df['기준_날짜'])
    pred_d['기준_날짜'] = pd.to_datetime(pred_d['기준_날짜'])

    aug = passes_df.merge(M_df, on=['기준_날짜','시간','정류장_순서'], how='left') \
                   .merge(pred_d, on=['기준_날짜','시간','정류장_순서'], how='left')
    aug['통과버스수'] = aug['통과버스수'].fillna(1.0)
    aug['예측_승차인원'] = aug['예측_승차인원'].fillna(0.0)
    aug['예측_하차인원'] = aug['예측_하차인원'].fillna(0.0)

    # 균등 분배
    aug['board_i']  = aug['예측_승차인원'] / aug['통과버스수']
    aug['alight_i'] = aug['예측_하차인원'] / aug['통과버스수']

    # 버스별 onboard
    aug = aug.sort_values(['bus_id','abs_min','정류장_순서']).reset_index(drop=True)
    cur = {}
    onboard = []
    for _, r in aug.iterrows():
        b = r['bus_id']
        x = cur.get(b, 0.0)
        x = max(0.0, x + float(r['board_i']) - float(r['alight_i']))
        if int(r['정류장_순서']) == S:  # 종점 리셋
            onboard.append(0.0)
            cur[b] = 0.0
        else:
            onboard.append(x)
            cur[b] = x
    aug['버스별_onboard'] = onboard

    # 시간×정류장 혼잡 메트릭
    avg = (aug.groupby(['기준_날짜','시간','정류장_순서'])['버스별_onboard']
              .mean().reset_index(name='버스당_탑승예측'))
    avg = avg.merge(M_df, on=['기준_날짜','시간','정류장_순서'], how='left')
    avg['통과버스수'] = avg['통과버스수'].fillna(0).astype(int)
    avg['총수요'] = avg['버스당_탑승예측'] * avg['통과버스수']

    # 혼잡 대수(정류장 단위): 총수요를 CAP로 나눠 필요한 총버스(stop,h)
    avg['필요_총버스'] = np.where(avg['총수요']>0, np.ceil(avg['총수요']/CAP).astype(int), 0)

    # 시간별 현재 출발대수(departures at first stop) = dep_map[(day,h)]
    dep_hour = pd.Series({h: int(dep_map.get((day, h), 0)) for h in range(24)})

    # 시간별 필요한 총버스(정류장별 필요의 최대값으로 정의; 합이 아니라 max로 중복 방지)
    need_by_hour = (avg.groupby('시간')['필요_총버스']
                      .max()  # 가장 많은 정류장을 기준으로 필요 총버스 정의
                      .reindex(range(24), fill_value=0))

    # 시간별 추가 필요 대수 = max(0, 필요_총버스(h) - dep_hour(h))
    add_by_hour = (need_by_hour - dep_hour).clip(lower=0)

    # 시간 요약 테이블
    hour_summary = pd.DataFrame({
        '시간': list(range(24)),
        '현재_출발대수': [int(dep_map.get((day, h), 0)) for h in range(24)],
        '필요_총버스_max정류장': need_by_hour.values,
        '추가_배차_시간': add_by_hour.values
    })

    # 참고용 정류장 단위 혼잡 스코어(버스당 탑승예측 - CAP)+
    avg['혼잡스코어'] = np.maximum(0.0, avg['버스당_탑승예측'] - CAP)

    # 최종: 정류장 단위 avg, 시간 단위 hour_summary 둘 다 반환
    return aug, M_df, avg, hour_summary

def greedy_reallocate(day, init_dep, max_iters=200):
    """
    init_dep: DataFrame[기준_날짜,시간,운행횟수]에서 특정 day만 추출한 것
    하루 총합 고정, 시간당 최소커버리지 보장, 시간별 증감은 SHIFT_LIMIT_PER_HOUR 이내
    """
    dep = { (day, int(r['시간'])): int(r['운행횟수']) for _, r in init_dep.iterrows() }
    total = sum(dep.values())

    # 초기 평가
    aug, M_df, avg, hour = simulate_passes(day, dep)
    best_dep = dep.copy()
    best_score = int(hour['추가_배차_시간'].sum())

    def find_donor_receiver(dep):
        # 여유 시간(혼잡스코어 낮고 dep>(MIN_COVERAGE))에서 1대 빼고,
        # 혼잡 큰 시간으로 1대 추가
        _, _, av, hour = simulate_passes(day, dep)
        # 시간별 추가 필요 대수(정류장 합이 아닌 시간별 max 기준)
        hour_need = hour.set_index('시간')['추가_배차_시간'].sort_values(ascending=False)
        # 여유 판단: 혼잡스코어 합이 작은 시간 우선 + 현재 출발대수>MIN_COVERAGE
        hour_slack = (av.groupby('시간')['혼잡스코어']
                        .sum()
                        .sort_values(ascending=True))  # 낮을수록 여유

        for h_need in hour_need.index.tolist():
            if hour_need.loc[h_need] <= 0:
                continue
            # 받을 곳
            recv = h_need
            # 줄 곳
            for h_don in hour_slack.index.tolist():
                if h_don == recv:
                    continue
                cur_dep = dep.get((day,h_don),0)
                if cur_dep <= MIN_COVERAGE:
                    continue
                # 혼잡 필요가 없는 시간 우선
                if hour_need.get(h_don, 0) > 0:
                    continue
                return h_don, recv
        return None, None

    it = 0
    while it < max_iters:
        it += 1
        h_don, h_recv = find_donor_receiver(dep)
        if h_don is None:
            break
        # 이동
        dep[(day,h_don)] -= 1
        dep[(day,h_recv)] = dep.get((day,h_recv),0) + 1

        # 총합 보존/제약 검사
        if sum(dep.values()) != total or dep[(day,h_don)] < MIN_COVERAGE:
            # 롤백
            dep[(day,h_don)] += 1
            dep[(day,h_recv)] -= 1
            break

        # 평가
        _, _, avg_new, hour_new = simulate_passes(day, dep)
        new_score = int(hour_new['추가_배차_시간'].sum())

        if new_score < best_score:
            best_score = new_score
            best_dep = dep.copy()
        else:
            # 개선 없으면 롤백하고 종료(또는 계속 탐색)
            dep[(day,h_don)] += 1
            dep[(day,h_recv)] -= 1
            break

    # 최종
    _, _, avg_best, hour_best = simulate_passes(day, best_dep)
    plan = pd.DataFrame([{'기준_날짜':day,'시간':h,'권장_출발대수':best_dep.get((day,h),0)} for h in range(24)])
    plan['기존_출발대수'] = plan['시간'].map(lambda h: int(dict(init_dep.set_index('시간')['운행횟수']).get(h,0)))
    plan['증감'] = plan['권장_출발대수'] - plan['기존_출발대수']

    return plan, best_score, avg_best

# ------------------------
# 실행: 하루씩 재배치 플랜 산출
# ------------------------
plans = []
scores = []
for d in days:
    init = now_departures[now_departures['기준_날짜'].dt.date == d][['시간','운행횟수']]
    if init.empty:
        continue
    plan, score, avg_best = greedy_reallocate(d, init)
    plan['총_추가배차_스코어(개선후)'] = score
    plans.append(plan)
    scores.append({'기준_날짜':d, '개선후_추가배차합':int(score)})

plans_df = pd.concat(plans, ignore_index=True) if plans else pd.DataFrame()
scores_df = pd.DataFrame(scores)

# 저장
OUT_PLAN = "/Users/minwoo/Desktop/softeer/data_engineering_course_materials/missions/final_project/result/172_정시성개선_권장출발대수_하루단위_탐욕안.csv"
OUT_SCORE = "/Users/minwoo/Desktop/softeer/data_engineering_course_materials/missions/final_project/result/172_정시성개선_혼잡스코어_요약.csv"
plans_df.to_csv(OUT_PLAN, index=False, encoding='utf-8-sig')
scores_df.to_csv(OUT_SCORE, index=False, encoding='utf-8-sig')
print("저장:", OUT_PLAN)
print("저장:", OUT_SCORE)