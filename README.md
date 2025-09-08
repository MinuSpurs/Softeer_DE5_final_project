# AI 기반 서울시 버스 운행 최적화 솔루션

## 1) Project Summary
-	서울시 버스의 혼잡도를 낮추고 시민 만족도를 높이는 B2G 데이터 프로덕트.
-	출퇴근길 수요–공급 불일치(빈차/만원차) 문제를 데이터 기반으로 해결합니다.



## 2) Problem Definition
-	Target: 서울시청 교통기획관 버스정책과
-	문제:
-	종로·도심 구간을 지나는 버스는 수요 변동을 반영하지 못한 고정 배차로 운행.
-	특정 시간에는 텅 빈 버스 → 세금·보조금 낭비.
-	다른 시간에는 만원 버스 → 시민 불편, 정시성 저하.



## 3) Solution

### (A) 수요 예측 (Demand Forecasting)
-	LSTM 모델을 활용해 정류장/시간대 단위의 단기(7일)·중기(14일) 승하차 예측.
-	요일·공휴일·날씨·행사 변수까지 반영하여 정확도 향상.
-	현재 프로토타입은 예측 + 재배치 계획 자동 산출까지 수행.

### (B) 최적화 시뮬레이션 (Optimization)
-	예측 결과를 기반으로 최소 차량으로 혼잡 완화하는 권장 출발 대수를 계산.
-	조건:
-	하루 총 대수 보존
-	보호 구간: 첫차+3시간, 막차−3시간은 이동 금지
-	혼잡 기준: 좌석 33명, 혼잡=좌석×1.5(≈50명) 초과
-	산출물: 여유 시간대 → 혼잡 시간대로 이동할 버스/대수 + 권장 헤드웨이(분)

### (C) 솔루션 제공 (대시보드/API)
-	대시보드:
-	일자별 요약 (현재 vs 권장 출발, 헤드웨이, 혼잡 히트맵)
-	정류장 상세 (예측 승·하차, 버스당 탑승, 혼잡도)
-	7일 예측 결과 제공
-	API: 운영·관제 시스템에 연동 가능 (예: /summary, /stop-detail, /forecast)



## 4) Architecture & Pipeline
-	EventBridge (매일 03:00): 오케스트레이터 Lambda 호출
-	수집 Lambda: 서울 열린데이터 API → S3 Bronze 저장
-	Spark on EC2: Bronze → Silver → Gold 변환 및 집계 → RDS 적재
-	LSTM (EC2/Lambda): 7일 예측 실행 → 결과를 RDS에 저장
-	API Gateway + Lambda: RDS 읽기 → 대시보드에 데이터 제공



## 5) Expected Impact
### 1.	시민 편익
-	혼잡 시간대 평균 탑승 감소 → 출퇴근길 앉아갈 확률 ↑
-	대기 시간 단축
### 2.	운영 효율
-	증차 없이 재배치만으로 혼잡 완화
-	공차·빈차 운행 최소화
### 3.	행정 효과
-	민원 감소, 정책 신뢰도 상승
### 4.	기술 확장성
-	Spark + LSTM 기반 자동화 파이프라인 → 타 노선·도시 확장 가능
-	AWS Serverless 아키텍처 → 운영 비용 최소화, 확장성 확보



## 6) Tech Stack
-	Data Source: 서울 열린데이터 광장 API, 기상 데이터(OpenWeather)
-	Infra: AWS (S3, RDS MySQL, EC2, Lambda, CloudWatch, API Gateway)
-	Processing: PySpark (ETL), Pandas
-	Model: TensorFlow/Keras (LSTM)
-	Dashboard: Streamlit (대시보드 시각화), API Gateway 연동
