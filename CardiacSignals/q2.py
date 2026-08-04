import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


CSV_PATH = "ecg_signals_preprocessed.csv"
OUTPUT_DIR = "outputs_q2"

RANDOM_STATE = 9
N_COMPONENTS_GMM = 3
CLASS_NAMES = ["N", "S", "V"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    X = df.iloc[:, :-1].values
    y_true = df.iloc[:, -1].values.astype(int)

    return X, y_true, df


def map_cluster_labels_to_true_labels(y_true, y_cluster):
    labels = np.arange(N_COMPONENTS_GMM)
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

    print("\n" + "=" * 70)
    print(experiment_name)
    print("=" * 70)
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


def save_results_table(results_list, save_path):
    rows = []

    for r in results_list:
        rows.append({
            "Experiment": r["experiment"],
            "Accuracy": r["accuracy"],
            "Macro Precision": r["precision_macro"],
            "Macro Recall": r["recall_macro"],
            "Macro F1-score": r["f1_macro"]
        })

    table = pd.DataFrame(rows)
    table.to_csv(save_path, index=False)

    print("\nSummary table:")
    print(table)

    return table


def main():
    X, y_true, df = load_data(CSV_PATH)
    x = X
    min_max_scaler = MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(x)

    # Train GMM-EM with all 83 features.
    gmm_all = GaussianMixture(
        n_components=N_COMPONENTS_GMM,
        covariance_type="full",
        random_state=RANDOM_STATE
    )

    gmm_all.fit(X_scaled)
    cluster_labels_all = gmm_all.predict(X_scaled)

    results_all, y_pred_all = evaluate_clustering(
        y_true,
        cluster_labels_all,
        "GMM-EM using all 83 normalised features"
    )

    plot_confusion_matrix(
        results_all["confusion_matrix"],
        "GMM-EM Confusion Matrix - All 83 Features",
        os.path.join(OUTPUT_DIR, "q2_gmm_all_features_confusion.png")
    )

    # Keep enough PCA components to explain at least 90% of variance.
    pca_full = PCA()
    pca_full.fit(X_scaled)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    retained_variance = cumulative_variance[n_components_90 - 1]

    pca = PCA(n_components=0.90, whiten=True)
    X_pca = pca.fit_transform(X_scaled)
    n_components_90 = X_pca.shape[1]
    retained_variance = np.sum(pca.explained_variance_ratio_)

    # Save the cumulative variance plot for PCA.
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
    plt.savefig(os.path.join(OUTPUT_DIR, "q2_pca_cumulative_variance.png"), dpi=300)
    plt.close()

    # Train GMM-EM with PCA-transformed features.
    gmm_pca = GaussianMixture(
        n_components=N_COMPONENTS_GMM,
        covariance_type="full",
        random_state=RANDOM_STATE
    )

    gmm_pca.fit(X_pca)
    cluster_labels_pca = gmm_pca.predict(X_pca)

    results_pca, y_pred_pca = evaluate_clustering(
        y_true,
        cluster_labels_pca,
        "GMM-EM after PCA with >90% explained variance"
    )

    plot_confusion_matrix(
        results_pca["confusion_matrix"],
        "GMM-EM Confusion Matrix - PCA Features",
        os.path.join(OUTPUT_DIR, "q2_gmm_pca_confusion.png")
    )

    # Save the result table.
    save_results_table(
        [results_all, results_pca],
        os.path.join(OUTPUT_DIR, "q2_gmm_summary.csv")
    )


if __name__ == "__main__":
    main()
