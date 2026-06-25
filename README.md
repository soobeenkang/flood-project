# 🌊 AI 기반 침수 예측 및 안전 대피 경로 안내 시스템

![Node.js](https://img.shields.io/badge/Node.js-Express-339933?logo=node.js&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-PostGIS%20%2B%20pgRouting-4169E1?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Cache-DC382D?logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

100m 격자 단위 AI 침수 예측과 IoT 실시간 수위 센서를 결합해, 침수 구역을 회피하는 안전 대피 경로를 안내하는 통합 재난 안전 플랫폼입니다.

🔗 **데모**: http://3.38.63.182:3000/

## 개요

기존의 행정구역(동 단위) 기준 재난 정보는 실제 지형·배수 특성을 반영하지 못해 미시적인 침수 위험 지역을 파악하기 어렵습니다. 이 프로젝트는 서울시를 100m 격자로 나눠 AI 모델로 침수를 예측하고, 자체 제작한 IoT 수위 센서로 실측 데이터를 보완해 정확도를 높였습니다. 사용자는 침수 구역을 우회하는 안전 대피 경로, 인근 대피소, 재난 행동 지침을 하나의 서비스에서 확인할 수 있습니다.

## 기능

- 1~6시간 후 100m 격자 단위 침수 여부를 예측하고 시간대별 히트맵으로 시각화
- ESP32 수위 센서의 실측값을 AI 예측보다 우선 반영하는 실시간 침수 판단
- PostGIS/pgRouting A* 알고리즘으로 침수 구역을 우회하는 대피 경로 산출 (최단 경로와 비교 가능)
- 위치 기반 인근 대피소 조회
- 관심 격자 이메일 알림 구독, 현재 위치 기준 실시간 침수 팝업 경고
- 정부 공식 재난 행동 지침 및 재난 문자 통합 조회

## 기술 스택

**Frontend**: React 19, Vite

**Backend**: Node.js(Express) API 서버, Python(데이터/ML) 서버

**Database**: PostgreSQL, PostGIS, pgRouting

**Cache & Messaging**: Redis, MQTT(Eclipse Mosquitto)

**AI/ML**: LightGBM, XGBoost, Optuna, scikit-learn

**Hardware**: ESP32, 수위 센서

**Infra**: Docker Compose, AWS EC2

## 아키텍처

```
ESP32 수위 센서 → MQTT(Mosquitto) → Node.js API → PostgreSQL/PostGIS, Redis
                                         ↑                ↓
                                   Python ML API ← 공공데이터(기상/지형/수위)
                                         ↓
                                   React 클라이언트
```

침수 판단은 **Redis(실시간 센서) → DB(센서 이력) → AI 예측값** 순으로 우선순위를 두어, 센서가 있는 지역은 실측값을, 없는 지역은 예측값을 사용합니다. 대피 경로는 이 침수 판단 결과를 바탕으로 도로망에서 A* 탐색을 수행해 산출합니다.

## 시작하기

### 요구사항

- Docker / Docker Compose
- Python 3.10
- Node.js

### 설치

```bash
git clone <repo-url>
cd flood-project-forclaude
cp .env.example .env   # 아래 표를 참고해 값 채우기
```

| 변수 | 설명 |
| --- | --- |
| `DB_USER` / `DB_PASSWORD` / `DB_NAME` / `DB_HOST` / `DB_PORT` | PostgreSQL 접속 정보 |
| `REDIS_URL` | Redis 접속 URL |
| `MQTT_URL` | MQTT 브로커 접속 URL |
| `PYTHON_API_URL` | Python ML API 주소 |
| `JWT_SECRET` | 인증 토큰 서명 키 |

### 실행

```bash
# 1. DB/Redis/MQTT/백엔드 컨테이너 기동
docker-compose up -d --build
```

| 서비스 | 포트 |
| --- | --- |
| PostgreSQL (PostGIS/pgRouting) | 5432 |
| Redis | 6379 |
| Mosquitto (MQTT / WebSocket) | 1883 / 9001 |
| Node API | 3000 |
| Python ML API | 8000 |

```bash
# 2. Python 환경 (데이터 처리/모델 학습 시)
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Frontend 개발 서버
cd frontend
npm install
npm run dev
```

## API

베이스 경로: `/api/v1`

| 엔드포인트 | 설명 |
| --- | --- |
| `GET /heatmap/grids` | 시간대별(now/1h/3h/6h) 침수 히트맵 그리드 조회 |
| `GET /heatmap/grids/all-horizons` | 전체 시간대 히트맵 한 번에 조회 |
| `GET /route/evacuation` | 침수 회피/최단 대피 경로 산출 |
| `GET /shelters` | 주변 대피소 조회 |
| `GET /sensors` | IoT 센서 최신 측정값 조회 |
| `POST /sensors/ingest` | 센서 측정값 수신 |
| `GET /location/flood-check` | 특정 위치 침수 여부 확인 |
| `GET /alerts` | 인근 재난 문자 조회 |
| `POST /subscriptions` | 관심 격자 알림 구독 |
| `GET /weather` | 위치 기반 기상 정보 조회 |

각 엔드포인트의 쿼리 파라미터는 해당 라우터 파일(`backend/node/src/api/v1/*/*.router.js`) 주석에 명시되어 있습니다.

## AI 모델

기상청·국토교통부·서울 열린데이터광장의 강수/지형/하수구 수위/침수 이력 데이터를 100m 격자 단위로 통합해 LightGBM, XGBoost 모델을 학습시켰습니다. 침수:비침수 비율이 178:1에 달하는 클래스 불균형을 완화하기 위해 주요 침수 시기(6~9월) 데이터로 한정하고, 비침수 데이터는 강우량 5mm 이상 조건으로 샘플링했습니다. `scale_pos_weight` 보정과 Optuna 하이퍼파라미터 최적화를 거쳐 테스트 데이터 기준 Precision·Recall 모두 0.9 이상을 확보했습니다.

## 폴더 구조

```
.
├── backend/
│   ├── node/             # API 서버 (경로 안내, 대피소, 알림, 센서 수신 등)
│   └── python-api/       # 데이터 수집·전처리, AI 모델 학습/추론
├── frontend/             # React 기반 웹/모바일 클라이언트
├── infra/mosquitto/      # MQTT 브로커 설정
├── data/                 # 원본/가공 데이터
└── docker-compose.yml
```

## 팀

엔코아 산학협력프로젝트 | 컴퓨터학부

| 이름 | 담당 업무 |
| --- | --- |
| 강수빈 | 데이터 처리, LightGBM 모델링, 경로 알고리즘 구축, 백엔드 |
| 신지우 | 데이터 처리, XGBoost 모델링, 아두이노, 백엔드 |
| 유예인 | 데이터 처리, LightGBM 모델링, 경로 알고리즘 구축, 백엔드 |
| 이재은 | 데이터 처리, XGBoost 모델링, 경로 알고리즘 구축, 프론트엔드 |
| 정소정 | 데이터 처리, XGBoost 모델링, 아두이노, 프론트엔드 |

## Acknowledgments

- 협력기관: 엔코아
- 한국정보처리학회(ASK2026) 게재 승인 논문: *"침수 피해 예측 AI 및 대피 경로 안내 시스템"*

## License

본 프로젝트는 학술적 목적의 산학협력(캡스톤 디자인) 결과물이며, 상업적 목적으로 사용되지 않습니다.
