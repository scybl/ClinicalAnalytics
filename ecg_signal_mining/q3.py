import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


CSV_PATH = "ecg_signals_preprocessed.csv"
OUTPUT_DIR = "outputs_q3"

N_CLUSTERS = 3
CLASS_NAMES = ["N", "S", "V"]

# Test two linkage methods.
LINKAGE_METHODS = ["average", "complete"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    X = df.iloc[:, :-1].values
    y_true = df.iloc[:, -1].values.astype(int)

    return X, y_true, df


def map_cluster_labels_to_true_labels(y_true, y_cluster):
    # Cluster labels have no class meaning, so test all label mappings.
    labels = np.arange(N_CLUSTERS)
    best_accuracy = -1
    best_mapping = None
    best_y_mapped = None

    for permutation in itertools.permutations(labels):
        cluster_to_label = {cluster_id: class_id for cluster_id, class_id in zip(labels, permutation)}
        y_mapped = np.array([cluster_to_label[c] for c in y_cluster])
        current_accuracy = accuracy_score(y_true, y_mapped)

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_mapping = cluster_to_label
            best_y_mapped = y_mapped

    return best_y_mapped, best_mapping


def evaluate_clustering(y_true, y_cluster, experiment_name):
    y_pred, mapping = map_cluster_labels_to_true_labels(y_true, y_cluster)

    cm = confusion_matrix(y_true, y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 80)
    print(experiment_name)
    print("=" * 80)
    print("Cluster-to-class mapping:")
    print(mapping)
    print("\nConfusion matrix:")
    print(cm)
    print(f"\nAccuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1-score:  {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    results = {
        "experiment": experiment_name,
        "mapping": mapping,
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1
    }

    return results, y_pred


def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")

    plt.xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    plt.yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar(label="Number of samples")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def results_to_dataframe(results_dict):
    rows = []

    for linkage, r in results_dict.items():
        rows.append({
            "Linkage": linkage,
            "Experiment": r["experiment"],
            "Accuracy": r["accuracy"],
            "Macro Precision": r["precision_macro"],
            "Macro Recall": r["recall_macro"],
            "Macro F1-score": r["f1_macro"]
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by="Accuracy", ascending=False)

    return df


def run_agglomerative_experiments(X_features, y_true, feature_type_name):
    all_results = {}

    for linkage in LINKAGE_METHODS:
        agglom = AgglomerativeClustering(
            n_clusters=N_CLUSTERS,
            linkage=linkage
        )

        agglom.fit(X_features)
        cluster_labels = agglom.labels_

        results, y_pred = evaluate_clustering(
            y_true,
            cluster_labels,
            f"Agglomerative clustering using {feature_type_name}, linkage={linkage}"
        )

        all_results[linkage] = results

    return all_results


def main():
    X, y_true, df = load_data(CSV_PATH)
    x = X
    min_max_scaler = MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(x)
    results_all = run_agglomerative_experiments(
        X_scaled,
        y_true,
        "all 83 normalised features"
    )

    summary_all = results_to_dataframe(results_all)
    summary_all_path = os.path.join(
        OUTPUT_DIR,
        "q3_agglomerative_all_linkage_summary.csv"
    )
    summary_all.to_csv(summary_all_path, index=False)

    print("\nSummary: Agglomerative clustering using all 83 features")
    print(summary_all)

    best_linkage_all = summary_all.iloc[0]["Linkage"]
    best_results_all = results_all[best_linkage_all]

    plot_confusion_matrix(
        best_results_all["confusion_matrix"],
        f"Agglomerative Confusion Matrix - All Features - {best_linkage_all}",
        os.path.join(
            OUTPUT_DIR,
            "q3_agglomerative_best_all_features_confusion.png"
        )
    )
    pca_full = PCA()
    pca_full.fit(X_scaled)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    retained_variance = cumulative_variance[n_components_90 - 1]

    pca = PCA(n_components=0.90, whiten=True)
    X_pca = pca.fit_transform(X_scaled)
    n_components_90 = X_pca.shape[1]
    retained_variance = np.sum(pca.explained_variance_ratio_)

    plt.figure(figsize=(7, 4))
    plt.plot(
        np.arange(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker="o"
    )
    plt.axhline(y=0.90, linestyle="--", label="90% variance threshold")
    plt.axvline(
        x=n_components_90,
        linestyle="--",
        label=f"{n_components_90} components"
    )
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "q3_pca_cumulative_variance.png"), dpi=300)
    plt.close()

    # Cluster with PCA-transformed features.
    results_pca = run_agglomerative_experiments(
        X_pca,
        y_true,
        f"PCA features ({n_components_90} components, {retained_variance:.4f} variance)"
    )

    summary_pca = results_to_dataframe(results_pca)
    summary_pca_path = os.path.join(
        OUTPUT_DIR,
        "q3_agglomerative_pca_linkage_summary.csv"
    )
    summary_pca.to_csv(summary_pca_path, index=False)

    print("\nSummary: Agglomerative clustering after PCA")
    print(summary_pca)

    best_linkage_pca = summary_pca.iloc[0]["Linkage"]
    best_results_pca = results_pca[best_linkage_pca]

    plot_confusion_matrix(
        best_results_pca["confusion_matrix"],
        f"Agglomerative Confusion Matrix - PCA - {best_linkage_pca}",
        os.path.join(
            OUTPUT_DIR,
            "q3_agglomerative_best_pca_confusion.png"
        )
    )

    # Save the final comparison table.
    final_comparison = pd.DataFrame([
        {
            "Feature representation": "All 83 normalised features",
            "Best linkage": best_linkage_all,
            "Accuracy": best_results_all["accuracy"],
            "Macro Precision": best_results_all["precision_macro"],
            "Macro Recall": best_results_all["recall_macro"],
            "Macro F1-score": best_results_all["f1_macro"]
        },
        {
            "Feature representation": f"PCA features ({n_components_90} components)",
            "Best linkage": best_linkage_pca,
            "Accuracy": best_results_pca["accuracy"],
            "Macro Precision": best_results_pca["precision_macro"],
            "Macro Recall": best_results_pca["recall_macro"],
            "Macro F1-score": best_results_pca["f1_macro"]
        }
    ])

    final_comparison_path = os.path.join(
        OUTPUT_DIR,
        "q3_agglomerative_final_comparison.csv"
    )
    final_comparison.to_csv(final_comparison_path, index=False)

    print("\nFinal Q3 comparison:")
    print(final_comparison)

    # Print class-level recall to support the confusion matrix discussion.
    cm_best = best_results_all["confusion_matrix"]
    class_recalls = cm_best.diagonal() / cm_best.sum(axis=1)

    print("\nClass-level recall based on best all-feature result:")
    for class_name, class_recall in zip(CLASS_NAMES, class_recalls):
        print(f"Recall for class {class_name}: {class_recall:.4f}")

    easiest_class = CLASS_NAMES[np.argmax(class_recalls)]
    hardest_class = CLASS_NAMES[np.argmin(class_recalls)]

    print(f"Easiest class to cluster according to recall: {easiest_class}")
    print(f"Most difficult class to cluster according to recall: {hardest_class}")


if __name__ == "__main__":
    main()
