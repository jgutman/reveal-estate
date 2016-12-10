import numpy as np
from sklearn.metrics import make_scorer, get_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    median_absolute_error, explained_variance_score

def accuracy_within_k_percent(y_true, y_pred, k = .10):
    deviation = np.abs(y_true - y_pred) / y_true.astype(float)
    accuracy = np.mean(deviation <= k)
    return accuracy

def accuracy_within_k_scorer(k):
    scorer = make_scorer(accuracy_within_k_percent, k=k)
    return scorer

def build_tuple_scorer(criterion_list):
    scorer_list = [parse_criterion_string(x) for x in criterion_list]
    tuple_score = lambda estimator, X, y: (score_fn(estimator, X, y)
        for score_fn in scorer_list)
    return tuple_score

def parse_criterion_string(criterion):
    criterion = criterion.lower()
    if criterion.startswith('accuracy'):
        k = criterion.split("_", 1)[1]
        k = float(k)
        if (k >= 1.0):
            k = k / 100.
        scorer = accuracy_within_k_scorer(k)
        return scorer
    elif (criterion == "mean_absolute_err" or criterion == "mae"):
        scorer = make_scorer(mean_absolute_error, greater_is_better = False)
        return scorer
    elif (criterion == "mean_squared_err" or criterion == "mse"):
        scorer = make_scorer(mean_squared_error, greater_is_better = False)
        return scorer
    elif (criterion == "median_absolute_err"):
        scorer = make_scorer(median_absolute_error, greater_is_better = False)
        return scorer
    elif (criterion == "var_explained"):
        scorer = make_scorer(explained_variance_score)
        return scorer
    else:
        try:
            scorer_from_string = get_scorer(criterion)
            return scorer_from_string
        except ValueError:
            print('could not parse criterion')
