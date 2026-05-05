#!/usr/bin/env python3
"""
Streamlit 前端应用
- 简历上传/粘贴
- 岗位类别预测
- Top5匹配岗位
- 技能缺口分析
- 词云展示
- 模型评估指标
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

# 配置
try:
    from config.settings import (
        MODELS_DIR,
        STREAMLIT_PAGE_ICON,
        STREAMLIT_TITLE,
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).parent
    MODELS_DIR = PROJECT_ROOT / "models"
    STREAMLIT_TITLE = "JD 关键词智能匹配系统"
    STREAMLIT_PAGE_ICON = "📊"

from core.matcher import match_resume
from core.preprocessor import CATEGORY_KEYWORDS

CLEAN_DATA_PATH = Path(__file__).parent / "data" / "processed" / "jobs_clean.json"

# 页面设置
st.set_page_config(page_title=STREAMLIT_TITLE, page_icon=STREAMLIT_PAGE_ICON, layout="wide")
st.title(f"{STREAMLIT_PAGE_ICON} {STREAMLIT_TITLE}")


@st.cache_data
def load_model_metrics() -> dict:
    """加载模型评估指标（从训练日志或直接计算）"""
    # 简单硬编码训练结果，也可以从日志文件解析
    return {
        "MultinomialNB": {"准确率": 0.7980, "F1-macro": 0.4452, "F1-weighted": 0.7365, "CV-F1": 0.7559},
        "LogisticRegression": {"准确率": 0.8182, "F1-macro": 0.4916, "F1-weighted": 0.7666, "CV-F1": 0.7915},
    }


@st.cache_data
def load_category_word_frequencies() -> dict[str, dict[str, int]]:
    """计算每个类别的词频，用于词云"""
    with open(CLEAN_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    from collections import Counter

    cat_words: dict[str, list[str]] = {}
    for r in data:
        cat = r.get("category", "未知")
        tokens = r.get("tokens", "")
        cat_words.setdefault(cat, []).extend(tokens.split())

    freq = {}
    for cat, words in cat_words.items():
        freq[cat] = dict(Counter(words).most_common(100))
    return freq


def generate_wordcloud(cat: str, freqs: dict[str, int]) -> np.ndarray:
    """生成词云图像数组"""
    wc = WordCloud(
        font_path=None,  # 使用默认字体，如有中文需求可指定中文字体路径
        width=400,
        height=300,
        background_color="white",
        max_words=50,
    )
    wc.generate_from_frequencies(freqs)
    return wc.to_array()


# ======================= 侧边栏 =======================
with st.sidebar:
    st.header("📝 简历输入")
    input_method = st.radio("选择输入方式", ["粘贴文本", "上传文件"])

    resume_text = ""
    if input_method == "粘贴文本":
        resume_text = st.text_area("请粘贴简历内容", height=300, placeholder="例如：熟悉Python、SQL、数据分析...")
    else:
        uploaded = st.file_uploader("上传简历文件", type=["txt", "md"])
        if uploaded:
            resume_text = uploaded.read().decode("utf-8")

    # FIX: st.button 没有 width 参数，改用 use_container_width
    analyze_btn = st.button("🔍 开始分析", type="primary", use_container_width=True)

    st.divider()
    st.info("""
    **使用说明：**
    1. 粘贴或上传简历文本
    2. 点击"开始分析"
    3. 查看岗位预测、匹配结果与技能缺口
    """)


# ======================= 主区域 =======================
if analyze_btn and resume_text.strip():
    with st.spinner("分析中，请稍候..."):
        result = match_resume(resume_text)

    # 主区域1：预测岗位类别
    st.subheader("1️⃣ 预测岗位类别")
    cat = result["predicted_category"]
    st.success(f"**{cat}**")

    # 主区域2：Top5匹配岗位
    st.subheader("2️⃣ Top5 最匹配岗位")
    matches = result["top_matches"]
    for i, m in enumerate(matches, 1):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"**{i}. {m['title']}** @ {m['company']}")
        with col2:
            st.caption(f"类别: {m['category']}")
        with col3:
            sim = m["similarity"]
            # FIX: st.progress 不使用 text 参数，避免 Streamlit Cloud 兼容性问题
            st.progress(min(sim * 3, 1.0))
            st.caption(f"相似度: {sim:.2%}")
        st.caption(f"[查看详情]({m['url']})")

    # 主区域3：技能缺口分析
    st.subheader("3️⃣ 技能缺口分析")
    gaps = result["skill_gap"]
    if gaps:
        cols = st.columns(min(len(gaps), 5))
        for i, skill in enumerate(gaps):
            with cols[i % 5]:
                st.error(f"❌ {skill}")
    else:
        st.success("✅ 核心技能覆盖完善，无明显缺口")

else:
    st.info('👈 请在侧边栏输入简历内容并点击「开始分析」')

# 模型评估与词云（无需简历输入也展示）
st.divider()

# 主区域5：模型评估指标
st.subheader("5️⃣ 模型评估指标")
metrics = load_model_metrics()
df_metrics = pd.DataFrame(metrics).T
# FIX: 不使用 width='stretch'，同时避免直接传入 Styler 对象
# 方案A：直接展示 DataFrame（推荐，Cloud 兼容性最好）
st.dataframe(df_metrics)
# 方案B（如果你一定要高亮，用 html table 替代）
# st.markdown(df_metrics.style.highlight_max(axis=0, color="green").to_html(), unsafe_allow_html=True)

# 主区域4：各类别核心技能词云
st.subheader("4️⃣ 各类别核心技能词云")
cat_freqs = load_category_word_frequencies()

# FIX: 用列表+cols动态渲染，避免最后一行空列导致 DOM 不匹配
if cat_freqs:
    cat_items = list(cat_freqs.items())
    n_cats = len(cat_items)
    cols_per_row = 3

    for row_start in range(0, n_cats, cols_per_row):
        row_cats = cat_items[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col_idx, (cat, freqs) in enumerate(row_cats):
            with cols[col_idx]:
                st.caption(f"**{cat}**")
                img = generate_wordcloud(cat, freqs)
                # FIX: 不使用 width='stretch'，改用 use_container_width
                st.image(img, use_container_width=True)
