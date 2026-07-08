"""Single-customer credit risk prediction with SHAP explanation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from credit_risk.features import engineer_features
from credit_risk.predict import predict_single

st.set_page_config(page_title="单客户预测", page_icon="🔍", layout="wide")
st.title("🔍 单客户违约风险预测")
st.caption("输入客户信贷信息，模型将实时返回违约概率与 SHAP 特征贡献解释。")

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("predict_form"):
    st.subheader("客户信息录入")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**基础信息**")
        age = st.number_input("年龄", min_value=18, max_value=109, value=45,
                               help="客户实际年龄（0 视为缺失）")
        monthly_income = st.number_input("月收入 ($)", min_value=0.0, max_value=100000.0,
                                          value=5400.0, step=100.0,
                                          help="月收入，留空或 0 视为缺失")
        n_dependents = st.number_input("赡养人数", min_value=0, max_value=20, value=0)

        st.markdown("**信贷账户**")
        n_open_lines = st.number_input("开放信贷额度/账户数", min_value=0, max_value=58, value=8)
        n_real_estate = st.number_input("房产贷款/信用额度数", min_value=0, max_value=54, value=1)

    with col2:
        st.markdown("**债务指标**")
        util = st.slider("无担保循环信用利用率", 0.0, 1.0, 0.3, 0.01,
                          help="信用卡等循环信贷余额 / 信用额度，>1 视为极度透支")
        debt_ratio = st.number_input("债务比率 (月债务/月收入)", min_value=0.0,
                                      max_value=5000.0, value=0.35, step=0.01)

        st.markdown("**历史逾期**")
        late_30 = st.number_input("30-59 天逾期次数", min_value=0, max_value=20, value=0)
        late_60 = st.number_input("60-89 天逾期次数", min_value=0, max_value=20, value=0)
        late_90 = st.number_input("90+ 天逾期次数",  min_value=0, max_value=20, value=0)

    model_choice = st.selectbox("选择模型", ["XGBoost（主模型）", "逻辑回归（基线）"])
    submitted = st.form_submit_button("🔮 开始预测", type="primary", use_container_width=True)

# ── Prediction & display ──────────────────────────────────────────────────────
if submitted:
    raw = {
        "RevolvingUtilizationOfUnsecuredLines":      float(util),
        "age":                                        float(age),
        "NumberOfTime30-59DaysPastDueNotWorse":       float(late_30),
        "DebtRatio":                                  float(debt_ratio),
        "MonthlyIncome":                              float(monthly_income) if monthly_income > 0 else None,
        "NumberOfOpenCreditLinesAndLoans":            float(n_open_lines),
        "NumberOfTimes90DaysLate":                   float(late_90),
        "NumberRealEstateLoansOrLines":              float(n_real_estate),
        "NumberOfTime60-89DaysPastDueNotWorse":      float(late_60),
        "NumberOfDependents":                         float(n_dependents),
    }
    model_key = "xgboost" if "XGBoost" in model_choice else "lr"

    with st.spinner("正在计算..."):
        try:
            features_df = engineer_features(raw)
            result      = predict_single(features_df, model=model_key)
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
            st.stop()

    prob      = result["probability"]
    risk      = result["risk_level"]
    thresh    = result["threshold"]
    shap_vals = result["shap_values"]

    st.divider()

    # ── Result summary ────────────────────────────────────────────────────
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        # Gauge chart
        risk_color = {"Low": "green", "Medium": "orange", "High": "red"}[risk]
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            delta={"reference": thresh * 100, "suffix": "% (阈值)"},
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "违约概率", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": risk_color, "thickness": 0.3},
                "bgcolor": "white",
                "steps": [
                    {"range": [0,  10], "color": "#d4edda"},
                    {"range": [10, 30], "color": "#fff3cd"},
                    {"range": [30, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": thresh * 100,
                },
            },
        ))
        gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(gauge, use_container_width=True)

        risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[risk]
        st.metric("风险等级", f"{risk_emoji} {risk}")
        decision_map = {"Low": "✅ 建议审批", "Medium": "⚠️ 人工复核", "High": "❌ 建议拒绝"}
        st.metric("决策建议", decision_map[risk])

    with res_col2:
        # SHAP waterfall
        st.subheader("特征贡献分析（SHAP）")
        st.caption("正值（红色）推高违约概率，负值（蓝色）降低违约概率")

        sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
        feats  = [x[0] for x in sorted_shap]
        values = [x[1] for x in sorted_shap]
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

        shap_fig = go.Figure(go.Bar(
            x=values,
            y=feats,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in values],
            textposition="outside",
        ))
        shap_fig.update_layout(
            xaxis_title="SHAP 值",
            yaxis={"autorange": "reversed"},
            height=420,
            margin=dict(l=10, r=60, t=10, b=40),
            plot_bgcolor="white",
            xaxis=dict(zeroline=True, zerolinecolor="black", zerolinewidth=1),
        )
        st.plotly_chart(shap_fig, use_container_width=True)

    with st.expander("查看完整 35 维特征向量"):
        st.dataframe(features_df.T.rename(columns={0: "值"}), use_container_width=True)
