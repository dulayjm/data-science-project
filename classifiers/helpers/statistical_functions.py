from statistics import mean, stdev
from typing import Union

from aenum import NamedTuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


Distribution: NamedTuple = NamedTuple("Distribution", "mean standard_deviation")


def calculate_base_statistics(predictions: list, ground_truth: list) -> tuple:
    if len(predictions) != len(ground_truth):
        raise ValueError("Length of predicted values and actual labels is not equal. "
                         "Proper comparison cannot be performed.")

    accuracy: float = accuracy_score(ground_truth, predictions)
    precision: float = precision_score(ground_truth, predictions, average='macro', zero_division=0)
    recall: float = recall_score(ground_truth, predictions, average='macro', zero_division=0)
    f_score: float = f1_score(ground_truth, predictions, average='macro', zero_division=0)

    return accuracy, precision, recall, f_score


def calculate_fold_statistics(*number_lists: list) -> list:
    distributions: list = []
    for number_list in number_lists:
        arithmetic_mean: float = mean(number_list)
        standard_deviation: float = stdev(number_list, arithmetic_mean)
        distributions.append(Distribution(arithmetic_mean, standard_deviation))
    return distributions


def display_base_statistics(seed: int, accuracy: float, precision: float, recall: float, f_score: float,
                            fold_number: Union[int, None] = None) -> None:
    if fold_number:
        displayed_string: str = f"Results (Seed: {seed}; Fold: {fold_number}):\n"
    else:
        displayed_string: str = f"Results (Seed: {seed}):\n"

    displayed_string += f"\t* Accuracy: {accuracy}\n"\
                        f"\t* Precision: {precision}\n"\
                        f"\t* Recall: {recall}\n"\
                        f"\t* F1 Score: {f_score}\n"

    print(displayed_string)


def display_fold_statistics(seed: int, fold_count: int, accuracy_distribution: Distribution,
                            precision_distribution: Distribution, recall_distribution: Distribution,
                            f1_distribution: Distribution) -> None:
    print(f"Overall Results (Seed: {seed}; Folds: {fold_count}):\n"
          f"\t* Accuracies: {accuracy_distribution.mean} +/- {accuracy_distribution.standard_deviation}\n"
          f"\t* Precisions: {precision_distribution.mean} +/- {precision_distribution.standard_deviation}\n"
          f"\t* Recalls: {recall_distribution.mean} +/- {recall_distribution.standard_deviation}\n"
          f"\t* F1 Scores: {f1_distribution.mean} +/- {f1_distribution.standard_deviation}")