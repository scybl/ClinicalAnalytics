import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import scipy.io as sio

def load_toe_dataset(mat_path: str) -> dict:
    data = sio.loadmat(mat_path)

    test_img = data["test_img"]
    gold_img = data["gold_img"][0]
    gen_impr = data["gen_impr"]
    crit_perc = data["crit_perc"]

    valid_mask = np.ones((20, 10), dtype=bool)

    for p in range(20):
        for v in range(10):

            img = test_img[p][v]

            g = gen_impr[p][v]
            c = crit_perc[p][v]

            # Mark sample as invalid if image or score is missing
            if (
                img is None
                or img.size == 0
                or g == -1
                or c == -1
            ):
                valid_mask[p, v] = False

    dataset = {
        "test_img": test_img,
        "gold_img": gold_img,
        "gen_impr": gen_impr,
        "crit_perc": crit_perc,
        "valid_mask": valid_mask
    }

    return dataset

def q1_i_correlation(dataset):

    gen_impr = dataset["gen_impr"]
    crit_perc = dataset["crit_perc"]
    valid_mask = dataset["valid_mask"]

    pearson_scores = []

    for view in range(10):

        general = []
        criteria = []

        for p in range(20):

            if not valid_mask[p, view]:
                continue

            general.append(gen_impr[p, view])
            criteria.append(crit_perc[p, view])

        general = np.array(general)
        criteria = np.array(criteria)

        r, _ = pearsonr(general, criteria)

        pearson_scores.append(r)

    return pearson_scores

def q1_ii_regression(dataset):

    gen_impr = dataset["gen_impr"]
    crit_perc = dataset["crit_perc"]
    valid_mask = dataset["valid_mask"]

    rmse_scores = []
    r2_scores = []
    predictions = {}

    for view in range(10):

        general = []
        criteria = []

        for p in range(20):

            if not valid_mask[p, view]:
                continue

            general.append(gen_impr[p, view])
            criteria.append(crit_perc[p, view])

        general = np.array(general)
        criteria = np.array(criteria)

        X = general.reshape(-1,1)
        y = criteria

        model = LinearRegression()
        model.fit(X,y)

        y_pred = model.predict(X)

        predictions[view] = (y, y_pred)

        rmse = np.sqrt(mean_squared_error(y,y_pred))
        r2 = r2_score(y,y_pred)

        rmse_scores.append(rmse)
        r2_scores.append(r2)

    return rmse_scores, r2_scores, predictions

def q1_iii_plot(predictions, r2_scores):

    best_views = np.argsort(r2_scores)[-3:][::-1]

    print("Best performing views:", best_views + 1)

    fig, axes = plt.subplots(1,3,figsize=(15,5))

    for i,view in enumerate(best_views):

        y_true, y_pred = predictions[view]

        axes[i].scatter(y_true, y_pred, s=60)

        axes[i].plot([0,100],[0,100],'r--',linewidth=2)

        axes[i].set_title(f"View {view+1}")
        axes[i].set_xlabel("True criteria percentage")
        axes[i].set_ylabel("Estimated criteria percentage")
        axes[i].grid(True)

    plt.suptitle("True vs Estimated Scores for Best Performing Views")

    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":

    dataset = load_toe_dataset("toe_image_quality.mat")

    # Q1(i)
    pearson_scores = q1_i_correlation(dataset)

    print("\nQ1(i) Pearson correlation")
    for i,r in enumerate(pearson_scores):
        print(f"View {i+1}: {r:.3f}")

    # Q1(ii)
    rmse_scores, r2_scores, predictions = q1_ii_regression(dataset)

    print("\nQ1(ii) Regression performance")

    for i in range(10):
        print(f"View {i+1}: RMSE={rmse_scores[i]:.3f}  R2={r2_scores[i]:.3f}")

    # Q1(iii)
    print("\nQ1(iii) Best view plots")
    q1_iii_plot(predictions, r2_scores)