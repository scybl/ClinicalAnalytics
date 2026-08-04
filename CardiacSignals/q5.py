import os
import itertools
import numpy as np
import pandas as pd


OUTPUT_DIR = "outputs_q5"
os.makedirs(OUTPUT_DIR, exist_ok=True)


try:
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError:
    apriori = None
    association_rules = None


CSV_PATH = "heart-statlog.csv"

MIN_SUPPORT = 0.25
MIN_LIFT = 1.15
MIN_CONVICTION = 1.5


def binarize_dataset(df):
    binary = pd.DataFrame(index=df.index)

    binary["age"] = (df["age"] > 50).astype(int)
    binary["sex"] = df["sex"].astype(int)
    binary["chest"] = (df["chest"] > 2.5).astype(int)
    binary["resting_blood_pressure"] = (df["resting_blood_pressure"] > 125).astype(int)
    binary["serum_cholestoral"] = (df["serum_cholestoral"] > 250).astype(int)
    binary["fasting_blood_sugar"] = df["fasting_blood_sugar"].astype(int)
    binary["resting_electrocardiographic_results"] = (
        df["resting_electrocardiographic_results"] != 0
    ).astype(int)
    binary["maximum_heart_rate_achieved"] = (
        df["maximum_heart_rate_achieved"] > 140
    ).astype(int)
    binary["exercise_induced_angina"] = df["exercise_induced_angina"].astype(int)
    binary["oldpeak"] = (df["oldpeak"] != 0).astype(int)
    binary["slope"] = (df["slope"] != 1).astype(int)
    binary["number_of_major_vessels"] = (df["number_of_major_vessels"] != 0).astype(int)
    binary["thal"] = (df["thal"] != 3).astype(int)
    binary["class"] = df["class"].map({"absent": 0, "present": 1}).astype(int)

    return binary


def encode_transactions(binary):
    item_names = {
        "age": {0: "age<=50", 1: "age>50"},
        "sex": {0: "sex=0", 1: "sex=1"},
        "chest": {0: "chest<=2.5", 1: "chest>2.5"},
        "resting_blood_pressure": {
            0: "resting_blood_pressure<=125",
            1: "resting_blood_pressure>125"
        },
        "serum_cholestoral": {
            0: "serum_cholestoral<=250",
            1: "serum_cholestoral>250"
        },
        "fasting_blood_sugar": {
            0: "fasting_blood_sugar=0",
            1: "fasting_blood_sugar=1"
        },
        "resting_electrocardiographic_results": {
            0: "resting_electrocardiographic_results=0",
            1: "resting_electrocardiographic_results!=0"
        },
        "maximum_heart_rate_achieved": {
            0: "maximum_heart_rate_achieved<=140",
            1: "maximum_heart_rate_achieved>140"
        },
        "exercise_induced_angina": {
            0: "exercise_induced_angina=0",
            1: "exercise_induced_angina=1"
        },
        "oldpeak": {0: "oldpeak=0", 1: "oldpeak!=0"},
        "slope": {0: "slope=1", 1: "slope!=1"},
        "number_of_major_vessels": {
            0: "number_of_major_vessels=0",
            1: "number_of_major_vessels!=0"
        },
        "thal": {0: "thal=3", 1: "thal!=3"},
        "class": {0: "class=absent", 1: "class=present"}
    }

    transaction_df = pd.DataFrame(index=binary.index)

    for column in binary.columns:
        for value, item_name in item_names[column].items():
            transaction_df[item_name] = binary[column].eq(value)

    return transaction_df


def local_apriori(transaction_df, min_support, use_colnames=True):
    n_rows = len(transaction_df)
    columns = list(transaction_df.columns)
    frequent_rows = []
    current_level = []

    for column in columns:
        itemset = frozenset([column])
        support = transaction_df[column].sum() / n_rows

        if support >= min_support:
            frequent_rows.append({"support": support, "itemsets": itemset})
            current_level.append(itemset)

    k = 2

    while current_level:
        candidates = set()

        for left, right in itertools.combinations(current_level, 2):
            candidate = left | right

            if len(candidate) == k:
                subsets_are_frequent = all(
                    frozenset(subset) in current_level
                    for subset in itertools.combinations(candidate, k - 1)
                )

                if subsets_are_frequent:
                    candidates.add(candidate)

        next_level = []

        for candidate in sorted(candidates, key=lambda x: tuple(sorted(x))):
            mask = transaction_df[list(candidate)].all(axis=1)
            support = mask.sum() / n_rows

            if support >= min_support:
                frequent_rows.append({"support": support, "itemsets": candidate})
                next_level.append(candidate)

        current_level = next_level
        k += 1

    frequent_itemsets = pd.DataFrame(frequent_rows)
    return frequent_itemsets


def local_association_rules(frequent_itemsets, metric="lift", min_threshold=1.0):
    support_lookup = {
        frozenset(row["itemsets"]): row["support"]
        for _, row in frequent_itemsets.iterrows()
    }

    rule_rows = []

    for itemset, itemset_support in support_lookup.items():
        if len(itemset) < 2:
            continue

        items = list(itemset)

        for size in range(1, len(items)):
            for antecedent_tuple in itertools.combinations(items, size):
                antecedents = frozenset(antecedent_tuple)
                consequents = itemset - antecedents

                antecedent_support = support_lookup[antecedents]
                consequent_support = support_lookup[consequents]
                confidence = itemset_support / antecedent_support
                lift = confidence / consequent_support
                leverage = itemset_support - antecedent_support * consequent_support

                if confidence == 1:
                    conviction = np.inf
                else:
                    conviction = (1 - consequent_support) / (1 - confidence)

                rule_rows.append({
                    "antecedents": antecedents,
                    "consequents": consequents,
                    "antecedent support": antecedent_support,
                    "consequent support": consequent_support,
                    "support": itemset_support,
                    "confidence": confidence,
                    "lift": lift,
                    "leverage": leverage,
                    "conviction": conviction
                })

    rules = pd.DataFrame(rule_rows)

    if rules.empty:
        return rules

    return rules[rules[metric] >= min_threshold].reset_index(drop=True)


def itemset_to_string(itemset):
    return ", ".join(sorted(list(itemset)))


def prepare_rules_for_csv(rules):
    csv_rules = rules.copy()
    csv_rules["antecedents"] = csv_rules["antecedents"].apply(itemset_to_string)
    csv_rules["consequents"] = csv_rules["consequents"].apply(itemset_to_string)
    return csv_rules


def run_association_analysis(transaction_df):
    if apriori is not None and association_rules is not None:
        frequent_itemsets = apriori(
            transaction_df,
            min_support=MIN_SUPPORT,
            use_colnames=True
        )
        rules = association_rules(
            frequent_itemsets,
            metric="lift",
            min_threshold=MIN_LIFT
        )
    else:
        frequent_itemsets = local_apriori(
            transaction_df,
            min_support=MIN_SUPPORT,
            use_colnames=True
        )
        rules = local_association_rules(
            frequent_itemsets,
            metric="lift",
            min_threshold=MIN_LIFT
        )

    rules = rules.sort_values(
        by=["lift", "conviction"],
        ascending=False
    ).reset_index(drop=True)

    conviction_rules = rules[
        rules["conviction"] > MIN_CONVICTION
    ].sort_values(
        by=["conviction", "lift"],
        ascending=False
    ).reset_index(drop=True)

    return frequent_itemsets, rules, conviction_rules


def filter_disease_rules(rules):
    disease_items = {"class=present", "class=absent"}

    disease_rules = rules[
        rules["consequents"].apply(lambda x: len(set(x) & disease_items) > 0)
    ].copy()

    disease_rules = disease_rules.sort_values(
        by=["conviction", "lift"],
        ascending=False
    ).reset_index(drop=True)

    return disease_rules


def main():
    df = pd.read_csv(CSV_PATH)

    binary = binarize_dataset(df)
    transaction_df = encode_transactions(binary)

    binary.to_csv(os.path.join(OUTPUT_DIR, "q5_binarized_data.csv"), index=False)
    transaction_df.to_csv(os.path.join(OUTPUT_DIR, "q5_transaction_data.csv"), index=False)

    frequent_itemsets, rules, conviction_rules = run_association_analysis(transaction_df)
    disease_rules = filter_disease_rules(conviction_rules)

    frequent_itemsets_csv = frequent_itemsets.copy()
    frequent_itemsets_csv["itemsets"] = frequent_itemsets_csv["itemsets"].apply(itemset_to_string)
    frequent_itemsets_csv = frequent_itemsets_csv.sort_values(
        by="support",
        ascending=False
    ).reset_index(drop=True)

    frequent_itemsets_csv.to_csv(
        os.path.join(OUTPUT_DIR, "q5_frequent_itemsets.csv"),
        index=False
    )
    prepare_rules_for_csv(rules).to_csv(
        os.path.join(OUTPUT_DIR, "q5_lift_rules.csv"),
        index=False
    )
    prepare_rules_for_csv(conviction_rules).to_csv(
        os.path.join(OUTPUT_DIR, "q5_conviction_rules.csv"),
        index=False
    )
    prepare_rules_for_csv(disease_rules).to_csv(
        os.path.join(OUTPUT_DIR, "q5_disease_rules.csv"),
        index=False
    )

    print("Dataset shape:")
    print(df.shape)
    print("\nFrequent itemsets:")
    print(frequent_itemsets_csv.head(15))
    print(f"\nNumber of frequent itemsets: {len(frequent_itemsets_csv)}")
    print(f"Number of rules with lift >= {MIN_LIFT}: {len(rules)}")
    print(f"Number of rules with conviction > {MIN_CONVICTION}: {len(conviction_rules)}")

    print("\nTop conviction rules:")
    print(
        prepare_rules_for_csv(conviction_rules)
        .head(10)[["antecedents", "consequents", "support", "confidence", "lift", "conviction"]]
    )

    print("\nTop disease-related conviction rules:")
    print(
        prepare_rules_for_csv(disease_rules)
        .head(10)[["antecedents", "consequents", "support", "confidence", "lift", "conviction"]]
    )


if __name__ == "__main__":
    main()
