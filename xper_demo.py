"""A small XPER-style performance attribution example."""

from itertools import combinations
from math import exp, factorial


FEATURES = ("stable_income", "low_debt", "no_overdue")
WEIGHTS = {"stable_income": 1.0, "low_debt": 0.9, "no_overdue": 1.2}
INTERCEPT = -1.5

# Each row contains three borrower features and the observed repayment result.
DATA = [
    ({"stable_income": 1, "low_debt": 1, "no_overdue": 1}, 1),
    ({"stable_income": 1, "low_debt": 1, "no_overdue": 0}, 1),
    ({"stable_income": 1, "low_debt": 0, "no_overdue": 1}, 1),
    ({"stable_income": 0, "low_debt": 1, "no_overdue": 1}, 1),
    ({"stable_income": 1, "low_debt": 0, "no_overdue": 0}, 0),
    ({"stable_income": 0, "low_debt": 1, "no_overdue": 0}, 0),
    ({"stable_income": 0, "low_debt": 0, "no_overdue": 1}, 0),
    ({"stable_income": 0, "low_debt": 0, "no_overdue": 0}, 0),
]


def sigmoid(value):
    return 1.0 / (1.0 + exp(-value))


def predict_probability(row, selected_features):
    score = INTERCEPT
    for feature in selected_features:
        score += WEIGHTS[feature] * row[feature]
    return sigmoid(score)


def performance(selected_features):
    """Return 1 minus the mean Brier loss; a larger value is better."""
    squared_errors = []
    for row, actual in DATA:
        predicted = predict_probability(row, selected_features)
        squared_errors.append((actual - predicted) ** 2)
    return 1.0 - sum(squared_errors) / len(squared_errors)


def shapley_contribution(feature):
    """Calculate one feature's exact Shapley contribution to performance."""
    others = [name for name in FEATURES if name != feature]
    contribution = 0.0
    feature_count = len(FEATURES)

    for subset_size in range(len(others) + 1):
        weight = (
            factorial(subset_size)
            * factorial(feature_count - subset_size - 1)
            / factorial(feature_count)
        )
        for subset in combinations(others, subset_size):
            without_feature = performance(subset)
            with_feature = performance(subset + (feature,))
            contribution += weight * (with_feature - without_feature)

    return contribution


def main():
    baseline = performance(())
    full_model = performance(FEATURES)
    contributions = {
        feature: shapley_contribution(feature) for feature in FEATURES
    }

    print(f"Baseline performance:   {baseline:.4f}")
    print(f"Full-model performance: {full_model:.4f}")
    print("Feature contributions:")
    for feature, value in contributions.items():
        print(f"  {feature:15s} {value:+.4f}")

    explained_gain = sum(contributions.values())
    total_gain = full_model - baseline
    print(f"Sum of contributions:   {explained_gain:+.4f}")
    print(f"Full minus baseline:    {total_gain:+.4f}")


if __name__ == "__main__":
    main()
