"""Batch credit risk prediction via CSV upload."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import io
import streamlit as st
import pandas as pd

from credit_risk.features import engineer_features, RAW_FEATURES
from credit_risk.predict import predict_batch

st.set_page_config(page_title="批量预测", page_icon="📋", layout="wide")
st.title("📋 批量违约风险预测")
st.caption("上传含有 10 项原始特征的 CSV，模型将输出每位客户的违约概率与风险等级。")

# ── Template download ─────────────────────────────────────────────────────────
template_df = pd.DataFrame(columns=RAW_FEATURES)
example_rows = [
    [0.5, 45, 0, 0.3, 5000, 8, 0, 1, 0, 2.0],
    [0.9, 28, 3, 1.5, 2000, 4, 1, 0, 2, 0.0],
    [0.1, 62, 0, 0.1, 8000, 12, 0, 2, 0, 1.0],
]
template_df = pd.DataFrame(example_rows, columns=RAW_FEATURES)
csv_template = template_df.to_csv(index=False)

st.download_button(
    "⬇️ 下载 CSV 模板",
    data=csv_template,
    file_name="credit_risk_template.csv",
    mime="text/csv",
)

st.info(
    f"**必须包含的列（顺序不限）：** {', '.join(RAW_FEATURES)}\n\n"
    "缺失值（MonthlyIncome / NumberOfDependents）可留空，"
    "年龄为 0 或逾期次数为 96/98 将自动处理。"
)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("上传客户数据 CSV", type=["csv"])

if uploaded:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"CSV 解析失败：{e}")
        st.stop()

    # Validate columns
    missing_cols = [c for c in RAW_FEATURES if c not in df_raw.columns]
    if missing_cols:
        st.error(f"缺少以下必要列：{missing_cols}")
        st.stop()

    st.success(f"成功读取 **{len(df_raw)}** 行客户数据")
    with st.expander("预览原始数据（前 5 行）"):
        st.dataframe(df_raw.head(), use_container_width=True)

    model_choice = st.selectbox("选择模型", ["XGBoost（主模型）", "逻辑回归（基线）"])
    model_key    = "xgboost" if "XGBoost" in model_choice else "lr"

    if st.button("🚀 开始批量预测", type="primary"):
        with st.spinner(f"正在为 {len(df_raw)} 位客户计算违约概率..."):
            try:
                features_df = engineer_features(df_raw[RAW_FEATURES])
                result_df   = predict_batch(features_df, model=model_key)
            except FileNotFoundError as e:
                st.error(f"❌ {e}")
                st.stop()

        # Attach original columns for context
        output_df = df_raw.copy()
        output_df["default_probability"] = result_df["default_probability"].values
        output_df["prediction"]          = result_df["prediction"].values
        output_df["risk_level"]          = result_df["risk_level"].values

        # ── Summary stats ────────────────────────────────────────────────
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总客户数",   len(output_df))
        c2.metric("🟢 低风险", (output_df["risk_level"] == "Low").sum())
        c3.metric("🟡 中风险", (output_df["risk_level"] == "Medium").sum())
        c4.metric("🔴 高风险", (output_df["risk_level"] == "High").sum())

        # ── Color-coded table ─────────────────────────────────────────────
        def highlight_risk(val):
            color_map = {"Low": "#d4edda", "Medium": "#fff3cd", "High": "#f8d7da"}
            return f"background-color: {color_map.get(val, '')}"

        display_cols = ["default_probability", "prediction", "risk_level"] + RAW_FEATURES[:5]
        styled = (
            output_df[display_cols]
            .style
            .applymap(highlight_risk, subset=["risk_level"])
            .format({"default_probability": "{:.2%}"})
        )
        st.dataframe(styled, use_container_width=True, height=400)

        # ── Download ──────────────────────────────────────────────────────
        csv_out = output_df.to_csv(index=False)
        st.download_button(
            "⬇️ 下载预测结果 CSV",
            data=csv_out,
            file_name="credit_risk_predictions.csv",
            mime="text/csv",
        )
