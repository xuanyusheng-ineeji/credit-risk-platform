# Credit Risk Assessment Platform

[![CI](https://github.com/YOUR_USERNAME/credit-risk-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/credit-risk-platform/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

**Language / 语言 / 언어**

[English](#english) · [中文](#中文) · [한국어](#한국어)

---

## English

A machine learning platform for personal credit default prediction, built on the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) Kaggle dataset (150,000 real credit records). Covers the full pipeline from EDA to a production-ready Streamlit web application.

### Features

| Feature | Description |
|---------|-------------|
| Single Prediction | Input 10 credit features and get real-time default probability, risk tier, and SHAP attribution chart |
| Batch Prediction | Upload CSV for bulk scoring; download results |
| Model Dashboard | ROC curve, feature importance, score distribution, risk tier breakdown |
| Dual Models | XGBoost (primary) + Logistic Regression (baseline), switchable |
| Docker Deployment | One-command startup, no local environment setup needed |
| Explainability | Native XGBoost TreeSHAP via `pred_contribs=True` |

### Model Performance

Evaluated on a stratified 20% validation split of 149,391 training samples.

| Metric | Logistic Regression | XGBoost (tuned) |
|--------|-------------------|-----------------|
| ROC-AUC | 0.8597 | **0.8659** |
| PR-AUC | 0.3813 | **0.4045** |
| Gini | 0.7193 | **0.7318** |
| KS | 0.5735 | **0.5850** |
| F1 | 0.3338 | **0.4591** |
| Recall | **0.7807** | 0.5200 |
| Precision | 0.2122 | **0.4110** |

> Class balance: 93.3% normal / 6.7% default. XGBoost uses `scale_pos_weight ≈ 13.9`.

### Project Structure

```
credit-risk-platform/
├── src/credit_risk/         # Installable Python package
│   ├── features.py          # Feature engineering (10 → 35 features)
│   └── predict.py           # Model loading & inference (with SHAP)
├── app/
│   ├── Home.py              # Streamlit home page
│   └── pages/
│       ├── 1_Single_Prediction.py
│       ├── 2_Batch_Prediction.py
│       └── 3_Dashboard.py
├── notebooks/               # Research & training (01 EDA → 06 comparison)
├── models/                  # Trained model files (.pkl, volume-mounted)
├── data/processed/          # Cleaned & engineered data (volume-mounted)
├── tests/                   # 21 unit tests + 9 integration tests
├── reports/model_comparison.csv
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

### Quick Start

**Option 1 — Local**

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-platform.git
cd credit-risk-platform

conda create -n credit-risk python=3.10
conda activate credit-risk

pip install -r requirements.txt
pip install -e .
```

Train models (only needed once, models/ is empty on first clone):

```
Run notebooks in order: 02 → 03 → 04 → 05
```

Start the app:

```bash
streamlit run app/Home.py
# Open http://localhost:8501
```

**Option 2 — Docker (recommended)**

```bash
docker compose up --build -d
# Open http://localhost:8501
```

> Models and data are volume-mounted. Ensure `models/` and `data/processed/` exist before starting.

### Running Tests

```bash
pip install -r requirements-dev.txt

# Unit tests — no model files required
pytest tests/test_features.py -v

# All tests (integration tests auto-skip if models absent)
pytest tests/ -v

# Coverage report
pytest tests/test_features.py --cov=credit_risk --cov-report=term-missing
```

### Feature Engineering

10 raw features → 35 engineered features:

| Step | Transformation | New Features |
|------|---------------|-------------|
| Missing flags | NaN / zero / 96-98 sentinel detection | 6 |
| Delinquency binary | ever_late_30/60/90 + total_late | 4 |
| Log transforms | MonthlyIncome / DebtRatio / Utilization | 3 |
| Age group | <30 / 30-45 / 45-60 / 60-75 / 75+ | 1 |
| Utilization group | 0-30% / 30-60% / 60-80% / 80-90% / 90%+ | 1 |
| Ratio & interaction | Monthly debt estimate, debt/credit lines, income/dependent, etc. | 8 |
| Late severity | 30d×1 + 60d×2 + 90d×4 + log | 2 |

### SHAP Explainability

XGBoost uses native `pred_contribs=True` (TreeSHAP) — avoids SHAP library incompatibility with XGBoost 2.x.

Key findings:
1. `total_late` / `util_x_late` / `late_severity_log` — historical delinquency is the strongest signal
2. `RevolvingUtilizationOfUnsecuredLines_log` — credit utilization reflects immediate financial stress
3. `MonthlyIncome_missing` — not providing income is itself a risk signal

### Tech Stack

| Category | Tools |
|----------|-------|
| Data | pandas, numpy |
| ML | scikit-learn, XGBoost |
| Explainability | XGBoost TreeSHAP |
| Visualization | matplotlib, seaborn, plotly |
| Web App | Streamlit |
| Testing | pytest, pytest-cov |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |

---

## 中文

基于机器学习的个人信贷违约预测系统，使用 [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) Kaggle 数据集（150,000 条真实信贷记录）。涵盖从 EDA 到生产级 Streamlit Web 应用的完整流程。

### 功能特性

| 功能 | 说明 |
|------|------|
| 单客户预测 | 输入 10 项信贷特征，实时输出违约概率 + 风险等级 + SHAP 归因图 |
| 批量预测 | CSV 上传批量打分，支持结果下载 |
| 模型仪表盘 | ROC 曲线、特征重要性、分数分布、风险等级分布 |
| 双模型 | XGBoost（主模型）+ 逻辑回归（基线），可切换 |
| Docker 部署 | 一条命令启动，无需本地环境配置 |
| 可解释性 | 原生 XGBoost TreeSHAP（`pred_contribs=True`） |

### 模型性能

在 149,391 条训练样本的 20% 分层验证集上评估：

| 指标 | 逻辑回归 | XGBoost（调优后） |
|------|----------|------------------|
| ROC-AUC | 0.8597 | **0.8659** |
| PR-AUC | 0.3813 | **0.4045** |
| Gini 系数 | 0.7193 | **0.7318** |
| KS 统计量 | 0.5735 | **0.5850** |
| F1 分数 | 0.3338 | **0.4591** |
| Recall（召回率） | **0.7807** | 0.5200 |
| Precision（精确率） | 0.2122 | **0.4110** |

> 类别比例：93.3% 正常 / 6.7% 违约。XGBoost 使用 `scale_pos_weight ≈ 13.9` 处理不平衡。

### 项目结构

```
credit-risk-platform/
├── src/credit_risk/         # 可安装 Python 包
│   ├── features.py          # 特征工程（10 → 35 个特征）
│   └── predict.py           # 模型加载 & 推理（含 SHAP）
├── app/
│   ├── Home.py              # Streamlit 主页
│   └── pages/
│       ├── 1_Single_Prediction.py   # 单客户预测
│       ├── 2_Batch_Prediction.py    # 批量预测
│       └── 3_Dashboard.py           # 模型仪表盘
├── notebooks/               # 研究 & 训练流程（01 EDA → 06 对比分析）
├── models/                  # 训练好的模型文件（.pkl，volume 挂载）
├── data/processed/          # 预处理后数据（volume 挂载）
├── tests/                   # 21 个单元测试 + 9 个集成测试
├── reports/model_comparison.csv
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

### 快速开始

**方式一：本地运行**

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-platform.git
cd credit-risk-platform

conda create -n credit-risk python=3.10
conda activate credit-risk

pip install -r requirements.txt
pip install -e .
```

训练模型（首次克隆时 models/ 为空，需运行一次）：

```
依次运行 Notebook：02 → 03 → 04 → 05
```

启动应用：

```bash
streamlit run app/Home.py
# 浏览器访问 http://localhost:8501
```

**方式二：Docker 部署（推荐）**

```bash
docker compose up --build -d
# 浏览器访问 http://localhost:8501

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

> 模型文件和数据通过 volume 只读挂载，不打入镜像。启动前请确保 `models/` 和 `data/processed/` 目录存在。

### 运行测试

```bash
pip install -r requirements-dev.txt

# 特征工程单元测试（无需模型文件，推荐优先运行）
pytest tests/test_features.py -v

# 全部测试（推理测试需要模型文件，否则自动跳过）
pytest tests/ -v

# 带覆盖率报告
pytest tests/test_features.py --cov=credit_risk --cov-report=term-missing
```

### 特征工程

原始 10 个特征经过以下步骤生成 35 个特征：

| 步骤 | 内容 | 新增特征数 |
|------|------|-----------|
| 缺失值标志 | MonthlyIncome/Dependents 为空、age=0、逾期字段 96/98 编码 | 6 |
| 逾期二值特征 | ever_late_30/60/90 + total_late | 4 |
| 对数变换 | MonthlyIncome / DebtRatio / Utilization | 3 |
| 年龄分组 | <30 / 30-45 / 45-60 / 60-75 / 75+ | 1 |
| 利用率分组 | 0-30% / 30-60% / 60-80% / 80-90% / 90%+ | 1 |
| 比率 & 交互特征 | 月债务估算、债务/信用额、收入/赡养人等 | 8 |
| 逾期严重程度 | 30d×1 + 60d×2 + 90d×4 + log | 2 |

### SHAP 可解释性

使用 XGBoost 原生 `pred_contribs=True`（绕过 SHAP 库与 XGBoost 2.x 的兼容性问题）：

- **正值（红色）**：推高违约概率
- **负值（蓝色）**：降低违约概率

关键发现：
1. `total_late` / `util_x_late` / `late_severity_log` — 历史逾期是最强信号
2. `RevolvingUtilizationOfUnsecuredLines_log` — 信用利用率反映即时财务压力
3. `MonthlyIncome_missing` — 不填收入本身是风险信号

### 技术栈

| 类别 | 工具 |
|------|------|
| 数据处理 | pandas, numpy |
| 机器学习 | scikit-learn, XGBoost |
| 可解释性 | XGBoost TreeSHAP |
| 可视化 | matplotlib, seaborn, plotly |
| Web 应用 | Streamlit |
| 测试 | pytest, pytest-cov |
| 容器化 | Docker, Docker Compose |
| CI/CD | GitHub Actions |

---

## 한국어

머신러닝 기반 개인 신용 부도 예측 플랫폼입니다. [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) Kaggle 데이터셋(실제 신용 기록 150,000건)을 사용하며, EDA부터 프로덕션 수준의 Streamlit 웹 앱까지 전체 파이프라인을 포함합니다.

### 주요 기능

| 기능 | 설명 |
|------|------|
| 단일 고객 예측 | 10개 신용 특징 입력 → 실시간 부도 확률 + 리스크 등급 + SHAP 귀인 차트 |
| 배치 예측 | CSV 업로드로 대량 스코어링, 결과 다운로드 지원 |
| 모델 대시보드 | ROC 곡선, 특징 중요도, 점수 분포, 리스크 등급 분포 |
| 듀얼 모델 | XGBoost (주 모델) + 로지스틱 회귀 (기준선), 전환 가능 |
| Docker 배포 | 명령어 하나로 실행, 로컬 환경 설정 불필요 |
| 설명 가능성 | 네이티브 XGBoost TreeSHAP (`pred_contribs=True`) |

### 모델 성능

149,391개 학습 샘플의 20% 층화 검증 세트 기준:

| 지표 | 로지스틱 회귀 | XGBoost (튜닝 후) |
|------|-------------|-----------------|
| ROC-AUC | 0.8597 | **0.8659** |
| PR-AUC | 0.3813 | **0.4045** |
| Gini | 0.7193 | **0.7318** |
| KS | 0.5735 | **0.5850** |
| F1 점수 | 0.3338 | **0.4591** |
| 재현율 (Recall) | **0.7807** | 0.5200 |
| 정밀도 (Precision) | 0.2122 | **0.4110** |

> 클래스 비율: 정상 93.3% / 부도 6.7%. XGBoost는 `scale_pos_weight ≈ 13.9`로 불균형 처리.

### 프로젝트 구조

```
credit-risk-platform/
├── src/credit_risk/         # 설치 가능한 Python 패키지
│   ├── features.py          # 특징 엔지니어링 (10 → 35개 특징)
│   └── predict.py           # 모델 로딩 & 추론 (SHAP 포함)
├── app/
│   ├── Home.py              # Streamlit 홈 페이지
│   └── pages/
│       ├── 1_Single_Prediction.py   # 단일 고객 예측
│       ├── 2_Batch_Prediction.py    # 배치 예측
│       └── 3_Dashboard.py           # 모델 대시보드
├── notebooks/               # 연구 & 학습 (01 EDA → 06 비교 분석)
├── models/                  # 학습된 모델 파일 (.pkl, 볼륨 마운트)
├── data/processed/          # 전처리된 데이터 (볼륨 마운트)
├── tests/                   # 단위 테스트 21개 + 통합 테스트 9개
├── reports/model_comparison.csv
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

### 빠른 시작

**방법 1 — 로컬 실행**

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-platform.git
cd credit-risk-platform

conda create -n credit-risk python=3.10
conda activate credit-risk

pip install -r requirements.txt
pip install -e .
```

모델 학습 (최초 클론 시 `models/` 가 비어 있는 경우 한 번만 실행):

```
노트북 순서대로 실행: 02 → 03 → 04 → 05
```

앱 실행:

```bash
streamlit run app/Home.py
# 브라우저에서 http://localhost:8501 접속
```

**방법 2 — Docker (권장)**

```bash
docker compose up --build -d
# 브라우저에서 http://localhost:8501 접속

# 로그 확인
docker compose logs -f

# 종료
docker compose down
```

> 모델 파일과 데이터는 읽기 전용 볼륨으로 마운트됩니다 (이미지에 포함되지 않음). 시작 전 `models/`와 `data/processed/` 디렉토리가 존재해야 합니다.

### 테스트 실행

```bash
pip install -r requirements-dev.txt

# 특징 엔지니어링 단위 테스트 (모델 파일 불필요, CI에 적합)
pytest tests/test_features.py -v

# 전체 테스트 (모델 없으면 통합 테스트 자동 스킵)
pytest tests/ -v

# 커버리지 리포트
pytest tests/test_features.py --cov=credit_risk --cov-report=term-missing
```

### 특징 엔지니어링

원시 10개 특징 → 35개 엔지니어링 특징:

| 단계 | 변환 내용 | 추가 특징 수 |
|------|----------|------------|
| 결측값 플래그 | NaN / 0 / 96-98 센티넬 감지 | 6 |
| 연체 이진 특징 | ever_late_30/60/90 + total_late | 4 |
| 로그 변환 | MonthlyIncome / DebtRatio / Utilization | 3 |
| 연령 그룹 | <30 / 30-45 / 45-60 / 60-75 / 75+ | 1 |
| 활용률 그룹 | 0-30% / 30-60% / 60-80% / 80-90% / 90%+ | 1 |
| 비율 & 상호작용 | 월 부채 추정액, 부채/신용한도, 수입/부양가족 등 | 8 |
| 연체 심각도 | 30일×1 + 60일×2 + 90일×4 + log | 2 |

### SHAP 설명 가능성

XGBoost 네이티브 `pred_contribs=True` 사용 (XGBoost 2.x와의 SHAP 라이브러리 비호환성 우회):

- **양수 값 (빨간색)**: 부도 확률 증가에 기여
- **음수 값 (파란색)**: 부도 확률 감소에 기여

주요 발견:
1. `total_late` / `util_x_late` / `late_severity_log` — 과거 연체 이력이 가장 강력한 신호
2. `RevolvingUtilizationOfUnsecuredLines_log` — 신용 활용률이 즉각적인 재정 압박 반영
3. `MonthlyIncome_missing` — 수입 미기재 자체가 리스크 신호

### 기술 스택

| 분류 | 도구 |
|------|------|
| 데이터 처리 | pandas, numpy |
| 머신러닝 | scikit-learn, XGBoost |
| 설명 가능성 | XGBoost TreeSHAP |
| 시각화 | matplotlib, seaborn, plotly |
| 웹 앱 | Streamlit |
| 테스트 | pytest, pytest-cov |
| 컨테이너화 | Docker, Docker Compose |
| CI/CD | GitHub Actions |

---

## License

MIT License
