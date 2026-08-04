import scipy.io as sio
import numpy as np
import cv2
import math
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# load  the data
def load_toe_dataset(mat_path: str) -> dict:

    data = sio.loadmat(mat_path)

    test_img = data["test_img"]
    gold_img = data["gold_img"][0]   # (1,10) → (10,)
    gen_impr = data["gen_impr"]
    crit_perc = data["crit_perc"]

    # initialise mask indicating valid samples
    valid_mask = np.ones((20, 10), dtype=bool)

    for p in range(20):
        for v in range(10):

            img = test_img[p][v]

            g = gen_impr[p][v]
            c = crit_perc[p][v]

            # mark sample as invalid if image or score is missing
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

# ECC, using default parameter
def compute_alignment(test, gold):
    test = test.astype(np.float32)
    gold = gold.astype(np.float32)

    sz = gold.shape

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    try:

        cc, warp_matrix = cv2.findTransformECC(
            gold,
            test,
            warp_matrix,
            warp_mode,
            criteria
        )
        test_aligned = cv2.warpAffine(
            test,
            warp_matrix,
            (sz[1], sz[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

        # translation
        tx = warp_matrix[0, 2]
        ty = warp_matrix[1, 2]
        translation = np.sqrt(tx**2 + ty**2)

        # rotation
        rotation_rad = math.atan2(
            warp_matrix[1, 0],
            warp_matrix[0, 0]
        )
        rotation_deg = np.degrees(rotation_rad)
        return rotation_deg, translation
    
    except cv2.error:
        return np.nan, np.nan


# Load dataset
dataset = load_toe_dataset("toe_image_quality.mat")

test_img = dataset["test_img"]
gold_img = dataset["gold_img"]
gen_impr = dataset["gen_impr"]
crit_perc = dataset["crit_perc"]
valid_mask = dataset["valid_mask"]

# parameters
warp_mode = cv2.MOTION_EUCLIDEAN

number_of_iterations = 500
termination_eps = 1e-10

criteria = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    number_of_iterations,
    termination_eps,
)

# Q4(i)
records = []

for p in range(20):

    for v in range(10):

        if not valid_mask[p, v]:
            continue

        test = test_img[p][v]
        gold = gold_img[v]

        rotation, translation = compute_alignment(test, gold)

        GI = gen_impr[p][v]
        CP = crit_perc[p][v]

        records.append({
            "participant": p + 1,
            "view": v + 1,
            "rotation": rotation,
            "translation": translation,
            "GI": GI,
            "criteria": CP
        })

df = pd.DataFrame(records)

print("\nRotation & Translation values")
print(df.head())

# Remove NaN rows (important)
df = df.dropna()


# Q4(ii)
experts = df[df["participant"] <= 7]
novices = df[df["participant"] > 7]

print("\nExpert vs Novice statistical test")

for view in range(1,11):

    exp_view = experts[experts["view"] == view]
    nov_view = novices[novices["view"] == view]

    r_stat, r_p = ttest_ind(
        exp_view["rotation"],
        nov_view["rotation"],
        equal_var=False
    )

    t_stat, t_p = ttest_ind(
        exp_view["translation"],
        nov_view["translation"],
        equal_var=False
    )

    print(f"\nView {view}")
    print(f"Rotation p-value: {r_p:.4f}")
    print(f"Translation p-value: {t_p:.4f}")


# Q4(iii)
results = []

for view in range(1,11):

    view_data = df[df["view"] == view]

    for x_var in ["rotation","translation"]:
        for y_var in ["criteria","GI"]:

            data = view_data[[x_var,y_var]].dropna()

            if len(data) < 3:
                continue

            X = data[[x_var]].values
            y = data[y_var].values

            model = LinearRegression()
            model.fit(X,y)

            y_pred = model.predict(X)

            rmse = np.sqrt(mean_squared_error(y,y_pred))
            r2 = r2_score(y,y_pred)

            results.append({
                "view":view,
                "independent":x_var,
                "dependent":y_var,
                "RMSE":rmse,
                "R2":r2
            })

results_df = pd.DataFrame(results)

print("\nRegression results")
print(results_df)

# Compute average performance per view
view_score = results_df.groupby("view").agg({
    "R2":"mean",
    "RMSE":"mean"
}).reset_index()

best_views = view_score.sort_values(
    ["R2","RMSE"],
    ascending=[False,True]
).head(3)

print("\nBest performing views")
print(best_views)


views = best_views["view"].astype(int).tolist()

combos = [
    ("rotation","criteria"),
    ("rotation","GI"),
    ("translation","criteria"),
    ("translation","GI")
]

#draw diagram
plt.figure(figsize=(12,14))

for r,(x_var,y_var) in enumerate(combos):

    for c,view in enumerate(views):

        view_data = df[df["view"] == view]
        data = view_data[[x_var,y_var]].dropna()

        if len(data) < 3:
            continue

        X = data[[x_var]].values
        y = data[y_var].values

        model = LinearRegression()
        model.fit(X,y)

        y_pred = model.predict(X)

        r2 = r2_score(y,y_pred)
        rmse = np.sqrt(mean_squared_error(y,y_pred))

        x_grid = np.linspace(X.min(),X.max(),200).reshape(-1,1)
        y_grid = model.predict(x_grid)

        index = r*3 + c + 1
        plt.subplot(4,3,index)

        plt.scatter(
            X,
            y,
            s=60,
            alpha=0.8,
            edgecolors="black"
        )

        plt.plot(
            x_grid,
            y_grid,
            color="red",
            linewidth=2
        )

        ylabel = "Criteria %" if y_var=="criteria" else "General impression"

        plt.xlabel(x_var.capitalize())
        plt.ylabel(ylabel)

        plt.title(
            f"View {view}\nR²={r2:.3f}, RMSE={rmse:.3f}, n={len(X)}",
            fontsize=9
        )

        plt.grid(True)

plt.suptitle(
    "Q4(iii) Linear Regression Results",
    fontsize=16
)

plt.tight_layout(rect=[0,0,1,0.96])
plt.show()