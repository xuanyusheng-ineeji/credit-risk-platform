"""Model performance dashboard."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
import joblib

from credit_risk.predict import load_models, _xgb_booster, _xgb_thresh, _lr_model, _lr_thresh

st.set_page_config(page_title="模型仪表盘", page_icon="📊", layout="wide")
st.title("📊 模型性能仪表盘")

_PROJ = Path(__file__).parent.parent.parent

# ── Load models (lazy) ────────────────────────────────────────────────────────
try:
    load_models(_PROJ / "models")
    import credit_risk.predict as _pred_mod
    booster    = _pred_mod._xgb_booster
    xgb_thresh = _pred_mod._xgb_thresh
    lr_model   = _pred_mod._lr_model
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()

# ── 1. Metrics comparison ─────────────────────────────────────────────────────
st.subheader("一、验证集指标对比")

cmp_path = _PROJ / "reports" / "model_comparison.csv"
if cmp_path.exists():
    cmp_df = pd.read_csv(cmp_path, index_col=0)
    st.dataframe(
        cmp_df.style.highlight_max(axis=0, color="#d4edda")
                    .format("{:.4f}"),
        use_container_width=True,
    )
else:
    st.warning("未找到 reports/model_comparison.csv，请先运行 notebook 06。")

st.divider()

# ── 2. Feature importance ─────────────────────────────────────────────────────
st.subheader("二、XGBoost 特征重要性（gain）")

try:
    scores = booster.get_score(importance_type="gain")
    imp_df = (
        pd.Series(scores)
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
        .rename(columns={"index": "feature", 0: "gain"})
    )
    fig_imp = px.bar(
        imp_df, x="gain", y="feature", orientation="h",
        color="gain", color_continuous_scale="Blues",
        labels={"gain": "Gain", "feature": "特征"},
        title="Top 20 特征（gain 重要性）",
    )
    fig_imp.update_layout(yaxis={"autorange": "reversed"}, height=520,
                          coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)
except Exception as e:
    st.warning(f"特征重要性加载失败：{e}")

st.divider()

# ── 3. Score distribution ─────────────────────────────────────────────────────
st.subheader("三、分数分布（验证集抽样）")

feat_path = _PROJ / "data" / "processed" / "train_features.csv"
if feat_path.exists():
    from sklearn.model_selection import train_test_split
    from credit_risk.features import FEATURE_ORDER

    @st.cache_data(show_spinner="加载验证集数据...")
    def load_validation_data():
        df = pd.read_csv(feat_path)
        X  = df[FEATURE_ORDER]
        y  = df["SeriousDlqin2yrs"]
        _, X_val, _, y_val = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
        sample = X_val.sample(min(5000, len(X_val)), random_state=42)
        dmat   = xgb.DMatrix(sample, feature_names=FEATURE_ORDER)
        proba  = booster.predict(dmat)
        y_s    = y_val.loc[sample.index]
        return proba, y_s.values

    xgb_proba, y_val = load_validation_data()

    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        fig_dist = go.Figure()
        for label, name, color in [(0, "正常客户", "steelblue"), (1, "违约客户", "tomato")]:
            mask = y_val == label
            fig_dist.add_trace(go.Histogram(
                x=xgb_proba[mask], name=name,
                opacity=0.65, nbinsx=50,
                marker_color=color, histnorm="probability density",
            ))
        fig_dist.add_vline(x=xgb_thresh, line_dash="dash", line_color="black",
                           annotation_text=f"阈值={xgb_thresh:.2f}")
        fig_dist.update_layout(
            title="XGBoost 违约概率分布",
            xaxis_title="预测违约概率",
            yaxis_title="密度",
            barmode="overlay",
            height=360,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with dist_col2:
        # ROC curve
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, _ = roc_curve(y_val, xgb_proba)
        auc = roc_auc_score(y_val, xgb_proba)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                      name=f"XGBoost (AUC={auc:.4f})",
                                      line=dict(color="steelblue", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      name="Random", line=dict(color="gray", dash="dash")))
        fig_roc.update_layout(
            title="ROC 曲线",
            xaxis_title="FPR", yaxis_title="TPR",
            height=360,
        )
        st.plotly_chart(fig_roc, use_container_width=True)
else:
    st.info("未找到 data/processed/train_features.csv，跳过分数分布图。")

st.divider()

# ── 4. Risk tier breakdown ────────────────────────────────────────────────────
if feat_path.exists():
    st.subheader("四、风险等级分布")
    bins   = [-0.001, 0.10, 0.30, 1.001]
    labels = ["🟢 Low", "🟡 Medium", "🔴 High"]
    tier   = pd.cut(xgb_proba, bins=bins, labels=labels)
    counts = tier.value_counts().reindex(labels)

    fig_pie = px.pie(
        values=counts.values,
        names=counts.index,
        title="验证集风险等级分布",
        color_discrete_map={"🟢 Low": "#28a745", "🟡 Medium": "#ffc107", "🔴 High": "#dc3545"},
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(height=360)
    st.plotly_chart(fig_pie, use_container_width=True)
