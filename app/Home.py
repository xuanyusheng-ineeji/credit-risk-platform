"""Credit Risk Platform — Home page."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st

st.set_page_config(
    page_title="Credit Risk Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💳 Credit Risk Assessment Platform")
st.markdown(
    "基于机器学习的个人信贷违约预测系统 · "
    "数据集：*Give Me Some Credit* (Kaggle) · "
    "150,000 条真实信贷记录"
)

st.divider()

# ── Key metrics ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("训练样本",     "149,391")
c2.metric("特征数量",     "35")
c3.metric("XGBoost AUC",  "0.866")
c4.metric("Gini 系数",    "0.732")
c5.metric("KS 统计量",    "≥ 0.45")

st.divider()

# ── Navigation cards ─────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔍 单客户预测")
    st.markdown(
        "输入客户的 10 项信贷特征，实时计算违约概率并输出 SHAP 特征贡献分析。"
    )
    st.page_link("pages/1_Single_Prediction.py", label="前往单客户预测 →")

with col2:
    st.subheader("📋 批量预测")
    st.markdown(
        "上传包含客户数据的 CSV 文件，批量计算违约概率，支持结果下载。"
    )
    st.page_link("pages/2_Batch_Prediction.py", label="前往批量预测 →")

with col3:
    st.subheader("📊 模型仪表盘")
    st.markdown(
        "查看模型评估指标、ROC/PR 曲线、特征重要性及分数分布等分析图表。"
    )
    st.page_link("pages/3_Dashboard.py", label="前往模型仪表盘 →")

st.divider()

# ── Model overview ───────────────────────────────────────────────────────────
with st.expander("ℹ️ 模型说明", expanded=False):
    st.markdown("""
**使用模型：XGBoost（主模型）+ 逻辑回归（基线）**

| 步骤 | 说明 |
|------|------|
| 数据清洗 | 修正年龄异常值、96/98 编码、缺失值处理、异常值截断 |
| 特征工程 | 对数变换、年龄分组、利用率分组、债务比率特征、逾期严重程度 |
| 类别不平衡 | `scale_pos_weight` 约 13.9（93.3% 正常 vs 6.7% 违约）|
| 模型调优 | Early Stopping + GridSearchCV |
| 可解释性 | XGBoost 原生 TreeSHAP（`pred_contribs=True`）|

**风险等级说明**

| 等级 | 违约概率 | 建议 |
|------|----------|------|
| 🟢 Low | < 10% | 可直接审批 |
| 🟡 Medium | 10% – 30% | 需人工复核 |
| 🔴 High | > 30% | 建议拒绝 |
    """)

st.caption("Credit Risk Platform · Built with Streamlit · XGBoost + Logistic Regression")
