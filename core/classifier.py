#!/usr/bin/env python3
"""
特征工程 + 模型训练模块
- TF-IDF向量化
- MultinomialNB / LogisticRegression 对比
- 交叉验证 + 模型保存
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB

# 配置（优先从settings读取，回退默认值）
try:
    from config.settings import (
        MODELS_DIR,
        RANDOM_STATE,
        TEST_SIZE,
        TFIDF_MAX_FEATURES,
        TFIDF_NGRAM_RANGE,
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    TFIDF_MAX_FEATURES = 3000
    TFIDF_NGRAM_RANGE = (1, 2)
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

CLEAN_DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "jobs_clean.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[list[str], list[str]]:
    """加载清洗后的数据，返回 (texts, labels)"""
    with open(CLEAN_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [r["tokens"] for r in data if r.get("tokens")]
    labels = [r["category"] for r in data if r.get("tokens")]
    return texts, labels


def train_and_evaluate():
    """完整训练与评估流程"""
    print("=" * 50)
    print("任务3: 特征工程 + 模型训练")
    print("=" * 50)

    # 1. 加载数据
    print("[1/5] 加载数据...")
    texts, labels = load_data()
    print(f"      样本数: {len(texts)}, 类别数: {len(set(labels))}")

    # 2. TF-IDF向量化
    print(f"[2/5] TF-IDF向量化 (max_features={TFIDF_MAX_FEATURES}, ngram_range={TFIDF_NGRAM_RANGE})...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)
    print(f"      特征维度: {X.shape}")

    # 3. 数据集划分
    print(f"[3/5] 划分数据集 (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    # 4. 训练模型
    print("[4/5] 训练模型...")
    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n  --- {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"  准确率 (Accuracy): {acc:.4f}")
        print(f"  F1-macro: {f1_macro:.4f}")
        print(f"  F1-weighted: {f1_weighted:.4f}")
        print("  Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # 5折交叉验证
        print(f"  5折交叉验证中...")
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1_weighted")
        print(f"  CV F1-weighted: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

    # 5. 选择最佳模型并保存
    print("\n[5/5] 保存最佳模型...")
    best_name = max(results, key=lambda k: results[k]["f1_weighted"])
    best_model = results[best_name]["model"]
    print(f"      最佳模型: {best_name}")
    print(f"      测试集F1-weighted: {results[best_name]['f1_weighted']:.4f}")

    model_path = MODELS_DIR / "best_classifier.pkl"
    vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"      模型保存至: {model_path}")
    print(f"      向量化器保存至: {vec_path}")

    # 返回评估摘要
    summary = {
        "best_model": best_name,
        "vectorizer_dim": X.shape,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "models": {
            name: {
                k: v for k, v in info.items() if k != "model"
            }
            for name, info in results.items()
        },
    }
    return summary


if __name__ == "__main__":
    train_and_evaluate()
