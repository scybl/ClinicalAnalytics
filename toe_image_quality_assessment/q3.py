import numpy as np
import matplotlib.pyplot as plt

import scipy.io as sio
from scipy.stats import pearsonr, entropy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Help function
def rmse_r2(y_true, y_pred):
    # trans to float type
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, r2

def count_nonzero_coef(model, thresh=0.01):
    coef = model.coef_.copy()
    return int(np.sum(np.abs(coef) >= thresh))

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

def mutual_information(img1, img2, bins=64):

    # Compute joint histogram of two images
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)

    # Convert histogram to joint probability distribution
    pxy = hist_2d / (np.sum(hist_2d) + 1e-12)

    # Marginal probability distributions
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    # Compute entropy terms
    Hx = entropy(px + 1e-12)
    Hy = entropy(py + 1e-12)
    Hxy = entropy(pxy.flatten() + 1e-12)

    return Hx + Hy - Hxy

def compute_similarity(test_img, gold_img, valid_mask):

    # Initialise similarity matrices
    SSI = np.full((20, 10), np.nan, dtype=float)
    MI  = np.full((20, 10), np.nan, dtype=float)
    CS  = np.full((20, 10), np.nan, dtype=float)

    for p in range(20):
        for v in range(10):

            # Skip invalid samples
            if not valid_mask[p, v]:
                continue

            img = test_img[p][v]
            gold = gold_img[v]

            # Ensure valid image data
            if img is None or getattr(img, "size", 0) == 0:
                continue
            if img.shape != gold.shape:
                continue

            # Compute Structural Similarity Index
            dr = float(img.max() - img.min())
            if dr <= 0:
                dr = 1.0
            SSI[p, v] = structural_similarity(img, gold, data_range=dr)

            # Compute Mutual Information
            MI[p, v]  = mutual_information(img, gold)

            # Compute Cosine Similarity between flattened images
            CS[p, v]  = cosine_similarity(img.reshape(1, -1), gold.reshape(1, -1))[0, 0]

    return SSI, MI, CS

def plot_top_regressions(results, top_k=3):
    # draw top models
    top_models = results[:top_k]

    fig, axes = plt.subplots(1, top_k, figsize=(15,5))

    for i, r in enumerate(top_models):

        ax = axes[i]

        x_v = r["x"]
        y_v = r["y"]
        model = r["model"]

        ax.scatter(x_v, y_v, alpha=0.6, label="Data")

        x_grid = np.linspace(x_v.min(), x_v.max(), 400).reshape(-1,1)

        y_grid = model.predict(x_grid)

        ax.plot(x_grid, y_grid, color="red", linewidth=2, label="Polynomial Fit")

        ylab = "Criteria %" if r["score"] == "crit_perc" else "General impression"

        ax.set_xlabel(r["metric"])
        ax.set_ylabel(ylab)

        ax.set_title(
            f"View {r['view']+1}\n"
            f"{r['metric']} → {r['score']}\n"
            f"R²={r['r2']:.3f}, RMSE={r['rmse']:.3f}"
        )

        ax.grid(True)
        ax.legend()

    plt.suptitle("Top Polynomial Regression Models")
    plt.tight_layout()
    plt.show()

def gaussian_design_matrix(x, centers, sigma):

    # Construct Gaussian basis functions
    x = np.asarray(x).reshape(-1, 1)
    centers = np.asarray(centers).reshape(1, -1)

    return np.exp(-((x - centers) ** 2) / (2 * sigma ** 2))

def run_polynomial_regression_ranking(SSI, MI, CS, gen_impr, crit_perc, degree=7, alpha=0.001):
    # calculate ii result
    metrics = [
        ("SSI", SSI),
        ("MI", MI),
        ("CS", CS)
    ]

    scores = [
        ("gen_impr", gen_impr),
        ("crit_perc", crit_perc)
    ]

    results = []

    for v in range(10): # for every view
        for m_name, m_mat in metrics: # for every 
            for s_name, s_mat in scores:

                x = m_mat[:, v]
                y = s_mat[:, v]

                mask = ~np.isnan(x) & (y != -1)

                x_v = x[mask].reshape(-1,1).astype(float)
                y_v = y[mask].astype(float)

                if len(x_v) < 3:
                    continue

                model = make_pipeline(
                    PolynomialFeatures(degree, include_bias=False),
                    StandardScaler(),
                    Ridge(alpha=alpha)
                )

                model.fit(x_v, y_v)

                y_hat = model.predict(x_v)

                rmse, r2 = rmse_r2(y_v, y_hat)

                results.append({
                    "view": v,
                    "metric": m_name,
                    "score": s_name,
                    "rmse": rmse,
                    "r2": r2,
                    "model": model,
                    "x": x_v,
                    "y": y_v
                })

    # R2 and RMSE sort
    results = sorted(
        results,
        key=lambda x: (-x["r2"], x["rmse"])
    )

    return results


# Load data
dataset = load_toe_dataset("toe_image_quality.mat")

test_img = dataset["test_img"]
gold_img = dataset["gold_img"]
gen_impr = dataset["gen_impr"]
crit_perc = dataset["crit_perc"]
valid_mask = dataset["valid_mask"]

# SSI, MI, CS
SSI, MI, CS = compute_similarity(test_img, gold_img, valid_mask)

# Q3(i) correlation (print only)
print("\nQ3(i) Correlation between similarity metrics (per view)\n")
for v in range(10):
    ssi = SSI[:, v]
    mi  = MI[:, v]
    cs  = CS[:, v]
    mask = ~np.isnan(ssi) & ~np.isnan(mi) & ~np.isnan(cs)

    r_ssi_mi, _ = pearsonr(ssi[mask], mi[mask])
    r_ssi_cs, _ = pearsonr(ssi[mask], cs[mask])
    r_mi_cs,  _ = pearsonr(mi[mask],  cs[mask])

    print(f"View {v+1}: SSI-MI={r_ssi_mi:.3f}, SSI-CS={r_ssi_cs:.3f}, MI-CS={r_mi_cs:.3f}")


# Q3(ii) Regularized Polynomial Regression (Modified with Scaling)
print("\nQ3(ii) Regularized Polynomial Regression\n")

results = run_polynomial_regression_ranking(
    SSI, MI, CS,
    gen_impr, crit_perc
)

print("\nTotal regressions evaluated:", len(results))
print("\nTop 10 regression models:\n")

for i, r in enumerate(results[:10]):

    print(
        f"{i+1:02d}. "
        f"View {r['view']+1} | "
        f"{r['metric']} → {r['score']} | "
        f"Polynomial degree=7 | "
        f"R²={r['r2']:.3f} | RMSE={r['rmse']:.3f}"
    )

plot_top_regressions(results, top_k=3)


# Q3(iii) Gaussian basis regression (SSI -> gen_impr)
print("\nQ3(iii) Gaussian basis regression (automatic model selection)\n")

# hyperparameter search space
M_list = range(2, 12)
alpha_list = [0.001, 0.01, 0.1, 1]

results = []

# Model selection for each view
for v in range(10):

    x = SSI[:, v]
    y = gen_impr[:, v]

    mask = ~np.isnan(x) & (y != -1)

    x_v = x[mask].astype(float)
    y_v = y[mask].astype(float)

    if len(x_v) < 3:
        continue

    best_r2 = -np.inf
    best_model = None
    best_params = None

    for M in M_list:

        centers = np.linspace(x_v.min(), x_v.max(), M)

        sigma = float((x_v.max() - x_v.min()) / M)
        if sigma <= 1e-12:
            sigma = 1.0

        Phi = gaussian_design_matrix(x_v, centers, sigma)

        for alpha in alpha_list:

            model = Ridge(alpha=alpha)
            model.fit(Phi, y_v)

            y_hat = model.predict(Phi)

            rmse, r2 = rmse_r2(y_v, y_hat)

            # keep best model for this view
            if r2 > best_r2:
                best_r2 = r2
                best_model = (centers, sigma, model)
                best_params = (M, alpha, rmse)

    results.append({
        "view": v,
        "r2": best_r2,
        "rmse": best_params[2],
        "M": best_params[0],
        "alpha": best_params[1],
        "model": best_model
    })


# Select Top 3 views
results = sorted(results, key=lambda x: x["r2"], reverse=True)

top3 = results[:3]

print("Top 3 views based on R2:")
for r in top3:
    print(
        f"View {r['view']+1} | "
        f"R2={r['r2']:.3f}, RMSE={r['rmse']:.3f}, "
        f"M={r['M']}, alpha={r['alpha']}"
    )


# Plot Top 3
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, r in enumerate(top3):

    v = r["view"]

    centers, sigma, model = r["model"]

    x = SSI[:, v]
    y = gen_impr[:, v]

    mask = ~np.isnan(x) & (y != -1)

    x_v = x[mask].astype(float)
    y_v = y[mask].astype(float)

    Phi = gaussian_design_matrix(x_v, centers, sigma)

    y_hat = model.predict(Phi)

    rmse, r2 = rmse_r2(y_v, y_hat)

    ax = axes[i]

    ax.scatter(x_v, y_v)

    x_grid = np.linspace(x_v.min(), x_v.max(), 300)

    Phi_g = gaussian_design_matrix(x_grid, centers, sigma)

    y_grid = model.predict(Phi_g)

    ax.plot(x_grid, y_grid)

    ax.set_xlabel("SSI")
    ax.set_ylabel("General impression")

    ax.grid(True)

    ax.set_title(
        f"View {v+1}\n"
        f"R2={r2:.3f}, RMSE={rmse:.3f}\n"
        f"M={r['M']}, alpha={r['alpha']}"
    )

plt.suptitle(
    "Q3(iii) Gaussian Basis Regression (SSI -> General impression): Top 3 views"
)

plt.tight_layout()

plt.show()