# PROJECT SUMMARY — Credit Risk Assessment Platform

> 面向开发者的完整项目导读，读完本文档可无需阅读源代码直接上手。

---

## 1. 项目概述

基于机器学习的个人信贷违约预测系统，使用 [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) Kaggle 数据集（150,000 条真实信贷记录）。

**完整技术栈：**

| 类别 | 工具 |
|------|------|
| 数据处理 | pandas >= 2.0, numpy >= 1.24 |
| 机器学习 | scikit-learn >= 1.3, XGBoost >= 2.0 |
| 可解释性 | XGBoost 原生 TreeSHAP (`pred_contribs=True`) |
| 可视化 | matplotlib, seaborn, plotly |
| Web 应用 | Streamlit >= 1.28 |
| 测试 | pytest, pytest-cov |
| 容器化 | Docker, Docker Compose |
| CI/CD | GitHub Actions |

---

## 2. 目录结构

```
credit-risk-platform/
├── src/
│   └── credit_risk/            # 可安装 Python 包（pip install -e .）
│       ├── __init__.py         # 导出核心 API
│       ├── features.py         # 特征工程 pipeline（10 → 35 个特征）
│       └── predict.py          # 模型加载 & 推理（含 SHAP）
├── app/
│   ├── Home.py                 # Streamlit 主页（概览 + 导航）
│   └── pages/
│       ├── 1_Single_Prediction.py   # 单客户实时预测
│       ├── 2_Batch_Prediction.py    # CSV 批量打分
│       └── 3_Dashboard.py           # 模型性能仪表盘
├── notebooks/
│   ├── 01_EDA.ipynb                 # 探索性分析
│   ├── 02_data_cleaning.ipynb       # 数据清洗
│   ├── 03_feature_engineering.ipynb # 特征工程
│   ├── 04_logistic_regression.ipynb # 逻辑回归基线
│   ├── 05_xgboost_model.ipynb       # XGBoost 训练 & 调参
│   └── 06_model_comparison_shap.ipynb # 模型对比 & SHAP 分析
├── models/                     # 训练好的模型文件（不进 Docker 镜像）
│   ├── logistic_model.pkl      # sklearn Pipeline (3.5 KB)
│   ├── xgboost_model.pkl       # XGBoost Booster (283 KB)
│   └── xgb_threshold.pkl       # 决策阈值 (117 B)
├── data/
│   ├── cs-training.csv         # Kaggle 原始数据
│   └── processed/              # 预处理后数据（56.6 MB）
│       ├── train_clean.csv     # 149,391 行，清洗后原始特征
│       ├── test_clean.csv      # 101,503 行，清洗后原始特征
│       └── train_features.csv  # 149,391 行，35 工程特征 + 标签
├── tests/
│   ├── conftest.py             # 模型存在性检测 fixture
│   ├── test_features.py        # 特征工程单元测试（21 个，无需模型）
│   └── test_predict.py         # 推理集成测试（9 个，需模型文件）
├── reports/
│   └── model_comparison.csv    # 模型性能指标汇总
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml              # 包配置 + pytest 配置
├── requirements.txt            # 生产依赖
├── requirements-dev.txt        # 开发依赖
└── README.md
```

---

## 3. 数据集与标签

| 项目 | 数值 |
|------|------|
| 训练样本数 | 149,391 |
| 测试样本数 | 101,503 |
| 目标变量 | `SeriousDlqin2yrs`（2 年内严重逾期） |
| 正常（0） | 93.3% |
| 违约（1） | 6.7% |
| 类别不平衡处理 | `scale_pos_weight ≈ 13.9`（XGBoost） |

---

## 4. 特征工程

### 4.1 原始特征（10 个）

| 特征名 | 含义 |
|--------|------|
| `RevolvingUtilizationOfUnsecuredLines` | 无担保信用利用率 [0, 1] |
| `age` | 年龄（岁） |
| `NumberOfTime30-59DaysPastDueNotWorse` | 30-59 天逾期次数 |
| `DebtRatio` | 负债比率（月债务 / 月收入） |
| `MonthlyIncome` | 月收入（美元） |
| `NumberOfOpenCreditLinesAndLoans` | 开放信用额度 & 贷款数 |
| `NumberOfTimes90DaysLate` | 90 天以上逾期次数 |
| `NumberRealEstateLoansOrLines` | 不动产贷款数 |
| `NumberOfTime60-89DaysPastDueNotWorse` | 60-89 天逾期次数 |
| `NumberOfDependents` | 赡养人数 |

### 4.2 工程化特征（35 个）

完整 `FEATURE_ORDER`（模型训练时的精确顺序，推理时须一致）：

```python
FEATURE_ORDER = [
    # 原始 10 个
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
    "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents",
    # 缺失值标志（6 个）
    "MonthlyIncome_missing", "NumberOfDependents_missing", "age_missing",
    "NumberOfTime30-59DaysPastDueNotWorse_missing",
    "NumberOfTime60-89DaysPastDueNotWorse_missing",
    "NumberOfTimes90DaysLate_missing",
    # 逾期二值特征（4 个）
    "ever_late_30", "ever_late_60", "ever_late_90", "total_late",
    # 对数变换（3 个）
    "MonthlyIncome_log", "DebtRatio_log",
    "RevolvingUtilizationOfUnsecuredLines_log",
    # 分组编码（2 个）
    "age_group_code", "util_group_code",
    # 比率 & 交互特征（8 个）
    "monthly_debt_est", "monthly_debt_est_log",
    "debt_per_credit_line", "income_per_dependent",
    "income_per_dependent_log", "credit_line_per_age",
    "real_estate_ratio", "util_x_late",
    # 逾期严重程度（2 个）
    "late_severity", "late_severity_log",
]
```

### 4.3 工程步骤（10 步，按顺序执行）

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | **缺失值标志** | `MonthlyIncome_missing`（isna）；`age_missing`（==0）；逾期字段 96/98 视为缺失 |
| 2 | **修复异常值** | `age==0` → NaN |
| 3 | **截断离群值** | Utilization clip [0,1]；DebtRatio clip upper=10640.54；Income clip upper=72817.42 |
| 4 | **填充缺失值** | age→52.0；MonthlyIncome→5400.0；Dependents→0.0；逾期→0.0 |
| 5 | **逾期二值特征** | `ever_late_X = (col > 0)`；`total_late = sum(三个逾期字段)` |
| 6 | **对数变换** | `log1p(MonthlyIncome/DebtRatio/Utilization)` |
| 7 | **年龄分组** | bins [0,30,45,60,75,200] → 编码 0-4 |
| 8 | **利用率分组** | bins [-0.001,0.3,0.6,0.8,0.9,10] → 编码 0-4 |
| 9 | **比率 & 交互** | 月债务估算、债务/信用额、收入/赡养人、信用额/年龄、不动产比率、利用率×逾期 |
| 10 | **逾期严重程度** | `30d×1 + 60d×2 + 90d×4` + log1p |

### 4.4 核心 API

```python
from credit_risk import engineer_features, FEATURE_ORDER, RAW_FEATURES

# 单行（dict 输入）
raw = {"age": 45, "MonthlyIncome": 5400, ...}
df_35 = engineer_features(raw)          # → DataFrame(1行 × 35列)

# 批量（DataFrame 输入）
df_35 = engineer_features(df_raw)       # → DataFrame(N行 × 35列)
```

---

## 5. 模型训练

### 5.1 逻辑回归（notebook 04）

- **Pipeline：** StandardScaler → LogisticRegression(C=0.1, class_weight="balanced")
- **阈值优化：** 最大化验证集 F1 分数
- **保存：** `models/logistic_model.pkl`，`models/lr_threshold.pkl`

### 5.2 XGBoost（notebook 05）

- **早停训练：** `XGBClassifier(scale_pos_weight=13.9, eval_metric="aucpr", early_stopping_rounds=50, base_score=0.5)`
- **GridSearchCV 调参：** `max_depth=[4,6]`, `learning_rate=[0.05,0.1]`, `n_estimators=[200,400,600]`, `subsample=[0.8,1.0]`, `colsample_bytree=[0.8,1.0]`
- **SHAP：** 使用 `booster.predict(DMatrix, pred_contribs=True)`（原生 TreeSHAP，绕开 SHAP 库 XGBoost 2.x 兼容性问题）
- **保存：** `models/xgboost_model.pkl`，`models/xgb_threshold.pkl`

### 5.3 已知兼容性问题及解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `ValueError: could not convert string to float: '[5E-1]'` | XGBoost 2.x 将 `base_score` 内部存储为 `'[5E-1]'`（带方括号），SHAP 库旧版 `TreeEnsemble.__init__` 调用 `float(...)` 失败 | 完全绕过 `shap.TreeExplainer`，改用 `booster.predict(dmat, pred_contribs=True)` |
| `AttributeError: 'XGBClassifier' object has no attribute 'use_label_encoder'` | 旧版 XGBoost 的 pickle 包含 `use_label_encoder`，新版改为 `@property` 会抛错 | 加载后执行 `xgb_raw.__dict__.pop('use_label_encoder', None)`，再用 `.get_booster()` 获取 Booster，所有预测走原生 API |

---

## 6. 模型性能

来自 `reports/model_comparison.csv`（验证集，分层 80/20 划分）：

| 模型 | ROC-AUC | PR-AUC | Gini | KS | F1 | Recall | Precision |
|------|---------|--------|------|----|----|--------|-----------|
| 逻辑回归 | 0.8597 | 0.3813 | 0.7193 | 0.5735 | 0.3338 | 0.7807 | 0.2122 |
| **XGBoost（调优后）** | **0.8659** | **0.4045** | **0.7318** | **0.585** | **0.4591** | 0.52 | **0.411** |

> XGBoost 在 ROC-AUC、PR-AUC、Gini、KS、F1、Precision 全面领先；逻辑回归 Recall 更高（更保守的违约识别）。

**SHAP 关键发现：**
1. `total_late` / `util_x_late` / `late_severity_log` — 历史逾期是最强信号
2. `RevolvingUtilizationOfUnsecuredLines_log` — 信用利用率反映即时财务压力
3. `MonthlyIncome_missing` — 不填收入本身即为风险信号

---

## 7. 推理 API

### 7.1 加载模型

```python
from credit_risk import load_models
load_models()       # 懒加载单例，可安全重复调用
```

模型路径默认为 `<项目根目录>/models/`，由 `predict.py` 中 `_MODELS_DIR` 自动推导。

### 7.2 单客户预测

```python
from credit_risk import engineer_features, predict_single

raw = {
    "RevolvingUtilizationOfUnsecuredLines": 0.3,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 5400,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0,
}
features_df = engineer_features(raw)
result = predict_single(features_df, model="xgboost")
```

**返回值结构：**

```python
{
    "probability":  0.0412,          # float [0, 1]
    "prediction":   0,               # int {0, 1}
    "risk_level":  "Low",            # str {'Low', 'Medium', 'High'}
    "threshold":    0.5,             # 决策阈值
    "shap_values": {                 # 35 个特征的 SHAP 贡献值
        "total_late": -0.23,
        "RevolvingUtilizationOfUnsecuredLines_log": 0.12,
        ...
    },
    "bias":         0.0714,          # SHAP 基准值（偏置项）
}
```

**风险等级映射：**

| 等级 | 概率区间 | 建议 |
|------|----------|------|
| Low | < 0.10 | 建议审批 |
| Medium | 0.10 – 0.30 | 人工复核 |
| High | ≥ 0.30 | 建议拒绝 |

### 7.3 批量预测

```python
from credit_risk import engineer_features, predict_batch

features_df = engineer_features(df_raw[RAW_FEATURES])
result_df = predict_batch(features_df, model="xgboost")
# 返回原始 DataFrame + 3 列：default_probability, prediction, risk_level
```

---

## 8. Streamlit 应用

### 8.1 启动

```bash
streamlit run app/Home.py
# 浏览器访问 http://localhost:8501
```

### 8.2 页面说明

#### Home.py — 项目概览
- 5 个关键指标：149,391 样本 / 35 特征 / AUC 0.866 / Gini 0.732 / KS ≥ 0.45
- 3 个导航卡片（`st.page_link`）

#### 1_Single_Prediction.py — 单客户预测
- **输入：** 10 个原始特征（Slider + number_input 混合表单）+ 模型选择
- **输出：**
  - Plotly Gauge 图（违约概率 + 阈值线 + 颜色分区）
  - 风险等级 emoji + 决策建议
  - SHAP 水平条形图（Top 12 特征，红色=推高风险，蓝色=降低风险）
  - 可展开的完整 35 维特征向量

#### 2_Batch_Prediction.py — 批量预测
- **输入：** CSV 模板下载 → 用户填写 → 上传（含列名验证）
- **输出：** 颜色编码的结果表（Low=绿，Medium=黄，High=红）+ CSV 下载

#### 3_Dashboard.py — 模型仪表盘
- 模型对比表（`model_comparison.csv`，高亮最优值）
- XGBoost 增益重要性 Top 20（`booster.get_score(importance_type="gain")`）
- 验证集分数分布直方图（正常 vs 违约叠加）
- ROC 曲线（sklearn 计算，AUC 标注）
- 风险等级饼图

---

## 9. 测试

### 9.1 运行测试

```bash
# 仅特征工程测试（无需模型文件，推荐 CI 使用）
pytest tests/test_features.py -v

# 全部测试（需要 models/ 目录存在）
pytest tests/ -v

# 带覆盖率
pytest tests/test_features.py --cov=credit_risk --cov-report=term-missing
```

### 9.2 测试覆盖范围

**test_features.py（21 个，全部通过，无需模型）**

| 分类 | 测试函数 |
|------|----------|
| 输出结构 | column_count, column_order, single_row, batch_input |
| 缺失值处理 | income_flag, income_fill, dependents_flag, age_zero_flag, age_zero_fill, late_96_flag, late_98_flag |
| 截断逻辑 | utilization_clip_high, utilization_clip_low |
| 特征逻辑 | ever_late_flags_zero/nonzero, total_late, late_severity, log_transforms_nonneg, age_group_codes, util_group_codes, no_nans_in_output |

**test_predict.py（9 个，模型不存在时自动 skip）**

probability_range, dict_keys, risk_level, prediction_binary, shap_feature_count, batch_shape, batch_proba_valid, lr_compatibility, high_risk_customer

---

## 10. Docker 部署

### 10.1 快速启动

```bash
docker compose up --build -d
# 访问 http://localhost:8501
```

### 10.2 架构说明

- **基础镜像：** `python:3.10-slim`
- **端口：** 8501
- **模型 & 数据通过 volume 挂载（只读），不打入镜像：**
  - `./models:/app/models:ro`
  - `./data/processed:/app/data/processed:ro`
  - `./reports:/app/reports:ro`
- **健康检查：** `curl http://localhost:8501/_stcore/health`（每 30s，超时 10s）
- **环境变量：**
  - `STREAMLIT_THEME_BASE=light`
  - `STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50`（MB）

---

## 11. CI/CD

文件：`.github/workflows/ci.yml`

**触发：** push → main/develop，PR → main

**Test Job（矩阵：Python 3.10 & 3.11）：**

```
checkout → setup-python → pip install deps → flake8 lint
→ pytest test_features → pytest test_predict（模型不存在自动跳过）
→ pytest --cov --cov-fail-under=70 → codecov 上传（仅 3.10）
```

**Docker Job（依赖 Test 成功）：**

```
checkout → setup buildx → docker build（不推送，GHA 缓存）
```

---

## 12. 包 API 导出

```python
from credit_risk import (
    engineer_features,   # 特征工程 pipeline
    FEATURE_ORDER,       # 35 特征的精确顺序列表
    RAW_FEATURES,        # 10 原始特征名列表
    load_models,         # 懒加载模型单例
    predict_single,      # 单客户推理（含 SHAP）
    predict_batch,       # 批量推理
)
```

---

## 13. 开发快速上手

```bash
# 1. 创建环境
conda create -n credit-risk python=3.10
conda activate credit-risk

# 2. 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .          # 以开发模式安装 src/credit_risk 包

# 3. 训练模型（首次使用）
# 按顺序运行 notebooks/02 → 03 → 04 → 05
# 或仅运行 04 + 05（02/03 产出的 processed 数据已存在）

# 4. 运行测试
pytest tests/test_features.py -v    # 快速验证特征工程

# 5. 启动应用
streamlit run app/Home.py
```

---

*最后更新：2026-07-08*
