#!/usr/bin/env python3
"""
项目全局配置
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
COOKIES_DIR = DATA_DIR / "cookies"

# 模型/输出目录
MODELS_DIR = PROJECT_ROOT / "models"

# 停用词表路径（从GitHub下载的中文停用词表）
STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"

# ML 配置
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Streamlit 配置
STREAMLIT_TITLE = "JD 关键词智能匹配系统"
STREAMLIT_PAGE_ICON = "📊"
