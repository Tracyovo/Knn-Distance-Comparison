import csv
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer
from tqdm import tqdm


def triangular_discrimination(x, y):
    """计算两个向量之间的三角判别距离。"""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    denominator = x + y
    denominator[denominator == 0] = 1e-10
    return np.sum((x - y) ** 2 / denominator)


def prepare_features(X_train, X_test):
    """按训练集拟合 MinMax 和 L1 归一化，避免数据泄露。"""
    scaler = MinMaxScaler()
    normalizer = Normalizer(norm="l1")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_norm = normalizer.fit_transform(X_train_scaled)
    X_test_norm = normalizer.transform(X_test_scaled)
    return X_train_norm, X_test_norm


def get_datasets(random_state=42):
    """构造课程论文可控规模的多数据集基准。"""
    datasets = []

    iris = load_iris()
    datasets.append(("Iris", iris.data, iris.target))

    wine = load_wine()
    datasets.append(("Wine", wine.data, wine.target))

    breast = load_breast_cancer()
    datasets.append(("BreastCancer", breast.data, breast.target))

    digits = load_digits()
    datasets.append(("Digits", digits.data, digits.target))

    X_syn, y_syn = make_classification(
        n_samples=800,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=3,
        class_sep=1.2,
        random_state=random_state,
    )
    datasets.append(("Synthetic3C", X_syn, y_syn))

    return datasets


def evaluate_dataset(X, y, k_values, splitter, dataset_name=""):
    """
    返回每个 k 在四种距离下的 Accuracy、Macro-F1 和耗时。
    四种距离：欧几里得、曼哈顿、余弦、三角判别。
    """
    metric_configs = [
        ("euc", "euclidean"),
        ("man", "manhattan"),
        ("cos", "cosine"),
        ("tri", triangular_discrimination),
    ]

    rows = []

    # 外层：k 值循环 + 进度条
    for k in tqdm(k_values, desc=f"数据集 {dataset_name}", unit="k"):
        acc = {name: [] for name, _ in metric_configs}
        f1 = {name: [] for name, _ in metric_configs}
        elapsed = {name: [] for name, _ in metric_configs}

        # 内层：50 折交叉验证（不需要进度条，太快）
        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_train, X_test = prepare_features(X_train, X_test)

            for metric_name, metric_param in metric_configs:
                knn = KNeighborsClassifier(
                    n_neighbors=k,
                    metric=metric_param,
                    algorithm="brute",
                )
                start = time.perf_counter()
                knn.fit(X_train, y_train)
                pred = knn.predict(X_test)
                elapsed[metric_name].append(time.perf_counter() - start)

                acc[metric_name].append(accuracy_score(y_test, pred))
                f1[metric_name].append(f1_score(y_test, pred, average="macro"))

        row = {"k": k}
        for metric_name, _ in metric_configs:
            row[f"acc_{metric_name}_mean"] = float(np.mean(acc[metric_name]))
            row[f"acc_{metric_name}_std"] = float(np.std(acc[metric_name], ddof=1))
            row[f"f1_{metric_name}_mean"] = float(np.mean(f1[metric_name]))
            row[f"time_{metric_name}_mean"] = float(np.mean(elapsed[metric_name]))
        row["acc_diff_tri_euc"] = float(np.mean(acc["tri"]) - np.mean(acc["euc"]))
        rows.append(row)

    return rows


def write_csv(path, rows, headers):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main():
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    show_plot = False

    random_state = 42
    k_values = [1, 3, 5, 7, 9]
    splitter = RepeatedStratifiedKFold(
        n_splits=10, n_repeats=5, random_state=random_state
    )

    datasets = get_datasets(random_state=random_state)
    detailed_rows = []
    summary_rows = []

    metric_names = ["euc", "man", "cos", "tri"]
    metric_labels = ["Euclidean", "Manhattan", "Cosine", "Triangular"]

    print("=== 多数据集 KNN 多距离度量对比实验 ===")
    print(f"数据集数量: {len(datasets)}")
    print(f"距离度量: {', '.join(metric_labels)}")
    print(f"评估协议: 10折 x 5次重复交叉验证, k={k_values}")

    for name, X, y in datasets:
        X = np.asarray(X)
        y = np.asarray(y)

        per_k_rows = evaluate_dataset(X, y, k_values, splitter, dataset_name=name)
        for row in per_k_rows:
            detailed_row = {"dataset": name, **row}
            detailed_rows.append(detailed_row)

        # 基于三角距离准确率选择最优 K（保持与旧实验一致的逻辑）
        best = max(per_k_rows, key=lambda item: item["acc_tri_mean"])
        k_best = best["k"]

        # 在最优 K 值下，重新获取所有度量的配对准确率用于统计检验
        acc_paired = {name: [] for name in metric_names}
        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_train, X_test = prepare_features(X_train, X_test)

            metric_configs = [
                ("euc", "euclidean"),
                ("man", "manhattan"),
                ("cos", "cosine"),
                ("tri", triangular_discrimination),
            ]
            for metric_name, metric_param in metric_configs:
                knn = KNeighborsClassifier(
                    n_neighbors=k_best,
                    metric=metric_param,
                    algorithm="brute",
                )
                knn.fit(X_train, y_train)
                pred = knn.predict(X_test)
                acc_paired[metric_name].append(accuracy_score(y_test, pred))

        # Wilcoxon 检验：欧氏 vs 三角（保持与旧实验对比）
        diff_tri_euc = np.array(acc_paired["tri"]) - np.array(acc_paired["euc"])
        _, p_tri_euc = wilcoxon(diff_tri_euc)

        summary = {
            "dataset": name,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(np.unique(y).shape[0]),
            "best_k_by_tri": int(k_best),
        }
        for mn in metric_names:
            summary[f"acc_{mn}_mean"] = float(np.mean(acc_paired[mn]))
        for mn in metric_names:
            # 从 per_k_rows 中找到 best k 对应的时间
            summary[f"time_{mn}_mean"] = best[f"time_{mn}_mean"]
        summary["wilcoxon_p_tri_euc"] = float(p_tri_euc)

        summary_rows.append(summary)

        # 控制台输出
        acc_str = " | ".join(
            f"{ml}={summary[f'acc_{mn}_mean']:.4f}"
            for mn, ml in zip(metric_names, metric_labels)
        )
        print(
            f"[{name}] best_k={k_best} | {acc_str} | "
            f"p_tri_euc={summary['wilcoxon_p_tri_euc']:.4f}"
        )

    # 构建 CSV 表头
    detailed_headers = ["dataset", "k"]
    for mn in metric_names:
        detailed_headers += [
            f"acc_{mn}_mean",
            f"acc_{mn}_std",
            f"f1_{mn}_mean",
            f"time_{mn}_mean",
        ]
    detailed_headers.append("acc_diff_tri_euc")

    summary_headers = [
        "dataset",
        "n_samples",
        "n_features",
        "n_classes",
        "best_k_by_tri",
    ]
    for mn in metric_names:
        summary_headers.append(f"acc_{mn}_mean")
    for mn in metric_names:
        summary_headers.append(f"time_{mn}_mean")
    summary_headers.append("wilcoxon_p_tri_euc")

    write_csv("knn_multimetric_detailed_results.csv", detailed_rows, detailed_headers)
    write_csv("knn_multimetric_summary_results.csv", summary_rows, summary_headers)

    # 绘制柱状图：四个度量 × 五个数据集
    dataset_names = [row["dataset"] for row in summary_rows]
    x = np.arange(len(dataset_names))
    n_metrics = len(metric_names)
    width = 0.2  # 每组柱子的总宽度
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    plt.figure(figsize=(14, 6))
    for i, (mn, ml) in enumerate(zip(metric_names, metric_labels)):
        acc_values = [row[f"acc_{mn}_mean"] for row in summary_rows]
        offset = (i - (n_metrics - 1) / 2) * width
        plt.bar(x + offset, acc_values, width, label=ml, color=colors[i], alpha=0.85)

    plt.xticks(x, dataset_names)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("不同数据集上四种距离度量的 KNN 分类准确率对比")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("knn_multimetric_accuracy.png", dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()

    print("\n结果文件已生成:")
    print("- knn_multimetric_detailed_results.csv")
    print("- knn_multimetric_summary_results.csv")
    print("- knn_multimetric_accuracy.png")


if __name__ == "__main__":
    main()
