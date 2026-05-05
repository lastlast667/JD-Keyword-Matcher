#!/usr/bin/env python3
"""
数据清洗与文本预处理模块
- 数据清洗、自动标注、去重、过滤
- jieba分词、去停用词
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import jieba

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "shixiseng_all.json"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "jobs_clean.json"
STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"

# 类别关键词加权映射
CATEGORY_KEYWORDS: Dict[str, Dict[str, float]] = {
    "数据分析": {
        "数据分析": 3.0, "数据分析师": 3.0, "sql": 2.0, "python": 1.5,
        "excel": 1.5, "报表": 2.0, "可视化": 2.0, "指标": 1.5, "监控": 1.0,
        "tableau": 2.0, "powerbi": 2.0, "数据处理": 1.5, "数据清洗": 1.5,
        "统计": 1.5, "分析": 1.0, "维度": 1.0, "漏斗": 1.5, "ab测试": 1.5,
        "业务分析": 2.0, "经营分析": 2.0, "财务分析": 1.5, "用户分析": 1.5,
    },
    "大数据开发": {
        "大数据": 2.0, "hadoop": 2.5, "spark": 2.5, "hive": 2.5, "flink": 2.5,
        "kafka": 2.0, "etl": 2.5, "数据仓库": 2.5, "数据湖": 2.0, "实时计算": 2.0,
        "离线": 1.5, "mapreduce": 2.0, "hbase": 2.0, "hdfs": 2.0, "yarn": 1.5,
        "数仓": 2.0, "数据开发": 2.0, "数据建模": 1.5, "调度": 1.0, "数据治理": 1.5,
        "doris": 2.0, "clickhouse": 2.0, "presto": 2.0, "impala": 2.0,
    },
    "BI工程师": {
        "bi": 3.0, "商业智能": 3.0, "仪表板": 2.0, "dashboard": 2.0,
        "报表开发": 2.5, "数据可视化": 2.0, "看板": 2.0, "power bi": 2.5,
        "finebi": 2.5, "帆软": 2.5, "水晶报表": 2.0, "cognos": 2.0,
        "bi工具": 2.0, "bi报表": 2.0, "bi平台": 2.0, "报表系统": 2.0,
        "多维分析": 1.5, "olap": 2.0, "cube": 1.5, "superset": 2.0,
    },
    "数据挖掘": {
        "机器学习": 2.5, "深度学习": 2.5, "算法": 2.0, "模型": 1.5, "特征工程": 2.5,
        "预测": 1.5, "分类": 1.5, "聚类": 1.5, "回归": 1.5, "nlp": 2.0,
        "推荐系统": 2.0, "数据挖掘": 3.0, "人工智能": 1.5, "神经网络": 2.0,
        "tensorflow": 2.0, "pytorch": 2.0, "sklearn": 2.0, "xgboost": 2.0,
        "自然语言处理": 2.0, "计算机视觉": 2.0, "cv": 1.5, "大模型": 1.5,
        "aigc": 1.5, "算法工程师": 2.5, "挖掘": 1.5,
    },
    "数据运营": {
        "运营": 2.0, "用户增长": 2.5, "增长": 2.0, "留存": 1.5, "转化": 1.5,
        "拉新": 1.5, "gmv": 2.0, "dau": 2.0, "mau": 1.5, "arpu": 1.5,
        "ltv": 1.5, "roi": 1.5, "投放": 1.5, "渠道": 1.0, "策略": 1.0,
        "活动": 1.0, "内容运营": 2.0, "产品运营": 2.0, "社群运营": 1.5,
        "新媒体": 1.0, "seo": 1.5, "sem": 1.5, "裂变": 1.5, "促活": 1.5,
    },
}

# 非技术岗过滤关键词
FILTER_KEYWORDS = [
    "数据录入", "数据标注", "内容审核", "客服", "电话销售", "地推",
    "文员", "行政", "前台", "收银", "仓管", "保洁", "保安", "司机",
    "纯打字", "抄写", "录入员", "审核员", "信息审核", "图片审核",
]

# IT自定义词典
CUSTOM_WORDS = [
    "大数据", "机器学习", "深度学习", "数据仓库", "数据湖", "数据治理",
    "Flask", "Django", "Hadoop", "Spark", "Hive", "Flink", "Kafka",
    "ETL", "SQL", "Python", "R语言", "Tableau", "PowerBI", "FineBI",
    "TensorFlow", "PyTorch", "Sklearn", "XGBoost", "LightGBM",
    "数据可视化", "数据挖掘", "特征工程", "推荐系统", "自然语言处理",
    "计算机视觉", "神经网络", "大模型", "AIGC", "AB测试", "漏斗分析",
    "用户画像", "数据建模", "实时计算", "离线计算", "数据清洗",
    "MapReduce", "HBase", "HDFS", "YARN", "Doris", "ClickHouse",
    "Presto", "Impala", "Superset", "Cognos", "OLAP", "Dashboard",
    "GMV", "DAU", "MAU", "ARPU", "LTV", "ROI", "SEO", "SEM",
    "BI工程师", "数据分析师", "数据开发", "算法工程师",
]


def clean_text(text: str) -> str:
    """清洗文本：去HTML标签、特殊符号、无意义换行"""
    if not text:
        return ""
    # 去HTML标签
    text = re.sub(r"<[^>]+>", " ", text)
    # 去特殊符号（保留中文、英文、数字、常用标点）
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s.,;:!?，。；：！？、\"'（）()]", " ", text)
    # 合并多余空格和换行
    text = re.sub(r"\s+", " ", text)
    # 去无意义换行，保留段落
    text = text.replace("\n", " ").strip()
    return text


def is_non_tech_job(text: str) -> bool:
    """判断是否为非技术岗"""
    text_lower = text.lower()
    for kw in FILTER_KEYWORDS:
        if kw in text_lower:
            return True
    return False


def auto_label(text: str) -> Tuple[str, float]:
    """
    基于关键词加权投票自动标注类别
    返回: (category, score)
    """
    text_lower = text.lower()
    scores: Dict[str, float] = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0.0
        for kw, weight in keywords.items():
            count = text_lower.count(kw.lower())
            score += count * weight
        scores[category] = score

    if not scores or max(scores.values()) == 0:
        return "数据分析", 0.0  # 默认类别

    best = max(scores, key=scores.get)
    return best, scores[best]


def merge_small_categories(records: List[dict], min_count: int = 30) -> List[dict]:
    """合并数据量不足的类别到最相近的大类"""
    from collections import Counter

    counts = Counter(r["category"] for r in records)
    print(f"[合并前] 类别分布: {dict(counts)}")

    # 合并映射：小类 -> 目标类
    merge_map = {
        "BI工程师": "数据分析",  # BI与数据分析最接近
        "数据挖掘": "数据分析",  # 数据挖掘也偏向分析
        "数据运营": "数据分析",  # 运营也合并到分析
    }

    for r in records:
        if counts[r["category"]] < min_count and r["category"] in merge_map:
            old = r["category"]
            r["category"] = merge_map[r["category"]]
            r["category_merged_from"] = old

    new_counts = Counter(r["category"] for r in records)
    print(f"[合并后] 类别分布: {dict(new_counts)}")

    # 如果还有不足30条的，继续合并
    for cat, cnt in list(new_counts.items()):
        if cnt < min_count and cat != "数据分析":
            for r in records:
                if r["category"] == cat:
                    r["category_merged_from"] = cat
                    r["category"] = "数据分析"

    final_counts = Counter(r["category"] for r in records)
    print(f"[最终] 类别分布: {dict(final_counts)}")
    return records


def deduplicate(records: List[dict]) -> List[dict]:
    """按 company + description前50字 去重"""
    seen = set()
    unique = []
    for r in records:
        company = r.get("company", "").strip()
        desc = r.get("job_description", "").strip()[:50]
        key = f"{company}_{desc}"
        if key not in seen:
            seen.add(key)
            unique.append(r)
    print(f"[去重] {len(records)} -> {len(unique)} 条")
    return unique


def load_stopwords(path: Path) -> set:
    """加载停用词表"""
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def add_custom_dict():
    """加载jieba自定义词典"""
    for word in CUSTOM_WORDS:
        jieba.add_word(word)


def tokenize(text: str, stopwords: set) -> str:
    """
    jieba分词 + 去停用词
    返回: 空格分隔的分词结果字符串
    """
    words = jieba.lcut(text)
    filtered = []
    for w in words:
        w = w.strip().lower()
        # 过滤停用词、单字（除部分重要单字）、纯数字、纯标点
        if not w or w in stopwords:
            continue
        if len(w) == 1 and w not in {"sql", "r", "c"}:
            continue
        if re.match(r"^\d+$", w):
            continue
        if re.match(r"^[^\u4e00-\u9fa5a-zA-Z0-9]+$", w):
            continue
        filtered.append(w)
    return " ".join(filtered)


def preprocess_pipeline() -> List[dict]:
    """
    完整预处理流水线
    """
    print("=" * 50)
    print("开始数据预处理")
    print("=" * 50)

    # 1. 读取原始数据
    print(f"[1/6] 读取原始数据: {RAW_DATA_PATH}")
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"      共 {len(raw_data)} 条")

    # 2. 清洗 + 过滤非技术岗
    print("[2/6] 清洗文本并过滤非技术岗...")
    cleaned = []
    for r in raw_data:
        desc = clean_text(r.get("job_description", ""))
        if not desc or len(desc) < 30:
            continue
        if is_non_tech_job(desc):
            continue
        r["job_description"] = desc
        cleaned.append(r)
    print(f"      清洗后: {len(cleaned)} 条")

    # 3. 去重
    print("[3/6] 去重...")
    cleaned = deduplicate(cleaned)

    # 4. 自动标注
    print("[4/6] 自动标注类别...")
    for r in cleaned:
        cat, score = auto_label(r["job_description"])
        r["category"] = cat
        r["category_score"] = round(score, 2)

    # 5. 合并小类别
    print("[5/6] 合并小类别...")
    cleaned = merge_small_categories(cleaned, min_count=30)

    # 6. jieba分词 + 去停用词
    print("[6/6] jieba分词与去停用词...")
    add_custom_dict()
    stopwords = load_stopwords(STOPWORDS_PATH)
    print(f"      停用词表: {len(stopwords)} 个")

    for r in cleaned:
        r["tokens"] = tokenize(r["job_description"], stopwords)

    # 保存清洗结果
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLEAN_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 清洗结果 -> {CLEAN_DATA_PATH} ({len(cleaned)} 条)")

    # 打印统计
    from collections import Counter
    counts = Counter(r["category"] for r in cleaned)
    print("\n[类别分布统计]")
    for cat, cnt in counts.most_common():
        print(f"  {cat}: {cnt} 条")

    return cleaned


if __name__ == "__main__":
    preprocess_pipeline()
