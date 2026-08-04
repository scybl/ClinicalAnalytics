import numpy as np
import scipy.io as sio

from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy, ttest_ind

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



# Mutual Information
def mutual_information(img1, img2, bins=64):

    # compute joint histogram of the two images
    hist_2d, _, _ = np.histogram2d(
        img1.ravel(),
        img2.ravel(),
        bins=bins
    )

    # normalise to obtain joint probability distribution
    pxy = hist_2d / np.sum(hist_2d)

    # marginal probability distributions
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    # compute entropy terms
    Hx = entropy(px + 1e-10)
    Hy = entropy(py + 1e-10)
    Hxy = entropy(pxy.flatten() + 1e-10)

    return Hx + Hy - Hxy


# compute similarity metrics
def compute_similarity(toe_data):

    test_img = toe_data["test_img"]
    gold_img = toe_data["gold_img"][0]

    results = []

    for participant in range(20):

        group = "expert" if participant < 7 else "novice"

        for view in range(10):

            img = test_img[participant][view]

            if img.size == 0:
                continue

            gold = gold_img[view]

            # compute similarity metrics
            img = np.squeeze(img)
            gold = np.squeeze(gold)

            # compute structural similarity (SSI)
            ssi = structural_similarity(
                img,
                gold,
                data_range=gold.max() - gold.min()
            )

            # compute structural similarity (SSI)
            mi = mutual_information(img, gold)

            # flatten images for cosine similarity
            v1 = img.flatten().reshape(1,-1)
            v2 = gold.flatten().reshape(1,-1)

            # compute cosine similarity
            cs = cosine_similarity(v1, v2)[0][0]

            results.append({
                "participant": participant,
                "view": view,
                "group": group,
                "ssi": ssi,
                "mi": mi,
                "cs": cs
            })

    return results

# top-3 similar images
def report_top3(results):

    metrics = ["ssi","mi","cs"]

    print("\nTop 3 most similar images for each view\n")

    for view in range(10):

        # collect results for current view
        view_data = [r for r in results if r["view"] == view]

        print(f"View {view+1}")

        for m in metrics:

            # sort participants according to similarity score
            sorted_view = sorted(
                view_data,
                key=lambda x: x[m],
                reverse=True
            )

            top3 = sorted_view[:3]

            participants = [
                f"P{r['participant']+1}"
                for r in top3
            ]

            print(f"{m.upper()} top3:", participants)

        print()


# Expert vs Novice hypothesis test
def expert_novice_test(results):

    metrics = ["ssi","mi","cs"]

    print("\nExpert vs Novice hypothesis test\n")

    for view in range(10):

        view_data = [r for r in results if r["view"] == view]

        print(f"View {view+1}")

        for m in metrics:

            # expert vs novice hypothesis test
            expert = [r[m] for r in view_data if r["group"]=="expert"]
            novice = [r[m] for r in view_data if r["group"]=="novice"]

            # perform two-sample t-test
            stat, p = ttest_ind(expert, novice)

            print(
                f"{m.upper()} | expert mean={np.mean(expert):.3f} "
                f"novice mean={np.mean(novice):.3f} "
                f"p-value={p:.4f}"
            )

        print()


if __name__ == "__main__":
    dataset = load_toe_dataset("toe_image_quality.mat")
    results = compute_similarity(dataset)
    report_top3(results)
    expert_novice_test(results)