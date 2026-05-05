#!/usr/bin/env python3
"""
简历-JD匹配模块
- 简历文本 -> TF-IDF -> 与全库JD计算余弦相似度
- Top5匹配岗位
- TOP10技能缺口分析
"""

import json
import re
from pathlib import Path

import jieba
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 配置
try:
    from config.settings import MODELS_DIR
except ImportError:
    MODELS_DIR = Path(__file__).parent.parent / "models"

CLEAN_DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "jobs_clean.json"

# 加载模型和向量化器
_classifier = None
_vectorizer = None
_jd_data = None
_jd_vectors = None

# 各岗位类别的核心技能词表（用于缺口分析）
CATEGORY_SKILLS: dict[str, list[str]] = {
    "数据分析": [
        "sql", "python", "excel", "tableau", "powerbi", "统计学",
        "数据可视化", "指标体系", "漏斗分析", "ab测试", "业务分析",
        "数据清洗", "etl", "hive", "spark", "pandas", "numpy",
    ],
    "大数据开发": [
        "hadoop", "spark", "hive", "flink", "kafka", "etl",
        "数据仓库", "数据建模", "hdfs", "mapreduce", "hbase",
        "实时计算", "离线计算", "数据治理", "doris", "clickhouse",
    ],
    "BI工程师": [
        "bi", "powerbi", "tableau", "finebi", "帆软", "数据可视化",
        "报表开发", "dashboard", "olap", "sql", "etl", "数据仓库",
        "多维分析", "superset", "水晶报表", "cognos",
    ],
    "数据挖掘": [
        "机器学习", "深度学习", "python", "算法", "模型",
        "特征工程", "tensorflow", "pytorch", "sklearn", "xgboost",
        "自然语言处理", "推荐系统", "神经网络", "大模型", "数据挖掘",
    ],
    "数据运营": [
        "sql", "excel", "数据分析", "用户增长", "留存", "转化",
        "ab测试", "漏斗分析", "用户画像", "活动策划", "gmv", "dau",
        "roi", "渠道投放", "内容运营", "策略",
    ],
}

# 所有技能词（去重）
ALL_SKILLS = list(set(
    skill for skills in CATEGORY_SKILLS.values() for skill in skills
))


def _load_models():
    """懒加载模型、向量化器和数据"""
    global _classifier, _vectorizer, _jd_data, _jd_vectors
    if _classifier is None:
        _classifier = joblib.load(MODELS_DIR / "best_classifier.pkl")
    if _vectorizer is None:
        _vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    if _jd_data is None:
        with open(CLEAN_DATA_PATH, "r", encoding="utf-8") as f:
            _jd_data = json.load(f)
        texts = [r["tokens"] for r in _jd_data]
        _jd_vectors = _vectorizer.transform(texts)


def _extract_skills(text: str) -> set[str]:
    """从文本中提取技能关键词"""
    text_lower = text.lower()
    found = set()
    for skill in ALL_SKILLS:
        if skill.lower() in text_lower:
            found.add(skill)
    return found


def predict_category(resume_text: str) -> str:
    """预测简历所属岗位类别"""
    _load_models()
    # 复用preprocessor的分词逻辑（简化版）
    try:
        from core.preprocessor import add_custom_dict, tokenize, load_stopwords, STOPWORDS_PATH
    except ImportError:
        from preprocessor import add_custom_dict, tokenize, load_stopwords, STOPWORDS_PATH
    add_custom_dict()
    stopwords = load_stopwords(STOPWORDS_PATH)
    tokens = tokenize(resume_text, stopwords)
    vec = _vectorizer.transform([tokens])
    pred = _classifier.predict(vec)[0]
    return pred


def match_resume(resume_text: str, top_k: int = 5) -> dict:
    """
    简历匹配主函数
    返回: {
        "predicted_category": str,
        "top_matches": [{title, company, similarity, url, category}],
        "skill_gap": [str],  # TOP10缺失技能
    }
    """
    _load_models()

    # 1. 分词
    try:
        from core.preprocessor import add_custom_dict, tokenize, load_stopwords, STOPWORDS_PATH
    except ImportError:
        from preprocessor import add_custom_dict, tokenize, load_stopwords, STOPWORDS_PATH
    add_custom_dict()
    stopwords = load_stopwords(STOPWORDS_PATH)
    resume_tokens = tokenize(resume_text, stopwords)
    resume_vec = _vectorizer.transform([resume_tokens])

    # 2. 计算余弦相似度
    sims = cosine_similarity(resume_vec, _jd_vectors).flatten()

    # 3. 取Top-K
    top_indices = np.argsort(sims)[::-1][:top_k]
    top_matches = []
    for idx in top_indices:
        jd = _jd_data[idx]
        # page_title 中可尝试提取职位名（正则提取可读字符）
        page_title = jd.get("page_title", "")
        # 尝试提取如 "BI实习生"、"数据分析师" 等可读片段
        readable = re.findall(r"[A-Za-z0-9\u4e00-\u9fa5]+", page_title)
        title_guess = "".join(readable[:3]) if readable else jd.get("category", "未知岗位")

        top_matches.append({
            "title": title_guess,
            "company": jd.get("company", ""),
            "similarity": round(float(sims[idx]), 4),
            "url": jd.get("detail_url", ""),
            "category": jd.get("category", ""),
        })

    # 4. 技能缺口分析
    resume_skills = _extract_skills(resume_text)
    # 以预测类别为准，也可以取top1匹配岗位类别
    predicted_category = predict_category(resume_text)
    target_skills = set(CATEGORY_SKILLS.get(predicted_category, []))
    missing = target_skills - resume_skills
    # 排序：按词频或重要性（这里简单按字母序，实际可加权）
    skill_gap = sorted(list(missing))[:10]

    return {
        "predicted_category": predicted_category,
        "top_matches": top_matches,
        "skill_gap": skill_gap,
    }


if __name__ == "__main__":
    # 简单测试
    test_resume = """
    熟悉Python、SQL、Pandas数据分析，使用过Tableau制作可视化报表，
    了解统计学基础和AB测试方法，有电商数据分析实习经验。
    """
    result = match_resume(test_resume)
    def safe_print(text):
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("utf-8", errors="ignore").decode("ascii", errors="ignore"))
    safe_print("预测类别: " + result["predicted_category"])
    safe_print("\nTop5匹配岗位:")
    for m in result["top_matches"]:
        safe_print(f"  [{m['similarity']:.4f}] {m['title']} @ {m['company']} ({m['category']})")
    safe_print("\n技能缺口: " + str(result["skill_gap"]))
