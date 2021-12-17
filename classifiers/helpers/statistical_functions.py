"""
This file contains various statistical functions used for evaluating the results of a model.
It also helps with the process of performing cross-validation.
Finally, it has multiple functions to display results, as well.
"""

from statistics import mean, stdev
from typing import Union

from aenum import NamedTuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


Distribution: NamedTuple = NamedTuple("Distribution", "mean standard_deviation")


def calculate_base_statistics(predictions: list, ground_truth: list) -> tuple:
    """
    This function takes in a list of predicted labels and a list of ground truth labels.
    It uses functions from scikit-learn to calculate a set of standard statistics for the results.
    Args:
        - predictions: a list of predicted labels of length N
        - ground_truth: a list of ground truth labels of length N
    Returns:
        - accuracy: a decimal (float) representing the accuracy of the predicted values to the ground truth values.
        - precision: a decimal (float) representing the macro-precision
        of the predicted values to the ground truth values.
        - recall: a decimal (float) representing the macro-recall of the predicted values to the ground truth values.
        - f_score: a decimal (float) representing the macro-averaged F1 score
        of the predicted values to the ground truth values.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Length of predicted values and actual labels is not equal. "
                         "Proper comparison cannot be performed.")

    accuracy: float = accuracy_score(ground_truth, predictions)
    precision: float = precision_score(ground_truth, predictions, average='macro', zero_division=0)
    recall: float = recall_score(ground_truth, predictions, average='macro', zero_division=0)
    f_score: float = f1_score(ground_truth, predictions, average='macro', zero_division=0)

    return accuracy, precision, recall, f_score


def calculate_fold_statistics(*number_lists: list) -> list:
    """
    This function takes in a series of lists of related values.
    For example, one could insert a list of accuracies, precisions, recalls, and F1 scores over folds.
    It outputs summative statistics--averages and standard deviations--for such values.
    Args:
        - number_lists: a series of lists of integers. There is no particular constraint for any list in relation
        to another list.
    Returns: a list of Distribution objects which contain both the arithmetic mean (float)
    and standard deviation (float) for a given numerical list.
    """
    distributions: list = []
    for number_list in number_lists:
        arithmetic_mean: float = mean(number_list)
        standard_deviation: float = stdev(number_list, arithmetic_mean)
        distributions.append(Distribution(arithmetic_mean, standard_deviation))
    return distributions


def display_base_statistics(seed: int, accuracy: float, precision: float, recall: float, f_score: float,
                            fold_number: Union[int, None] = None) -> None:
    """
    This method prints out a series of statistics to the console.
    In particular, it is meant to display the outcome of a singular fold or a singular test for some model.
    Args:
        - seed: the random seed used in the process of creating the data (int).
        - accuracy: the accuracy of the model (float).
        - precision: the precision of the model (float).
        - recall: the recall of the model (float).
        - f_score: the F1 Score of the model (float).
        - fold_number: if folds were used, the current fold number during testing (int, if used).
    """
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
    """
    This method prints out a series of statistics to the console.
    In particular, it is meant to display the outcome of a series of folds for some model.
    Args:
        - seed: the random seed used in the process of creating the data (int).
        - fold_count: the number of folds used for testing the model (int).
        - accuracy_distribution: the mean and standard deviation of accuracies for the model (Distribution).
        - precision_distribution: the mean and standard deviation of precisions for the model (Distribution).
        - recall_distribution: the mean and standard deviation of recalls for the model (Distribution).
        - f1_distribution: the mean and standard deviation of F1 scores for the model (Distribution).
    """
    print(f"Overall Results (Seed: {seed}; Folds: {fold_count}):\n"
          f"\t* Accuracies: {accuracy_distribution.mean} +/- {accuracy_distribution.standard_deviation}\n"
          f"\t* Precisions: {precision_distribution.mean} +/- {precision_distribution.standard_deviation}\n"
          f"\t* Recalls: {recall_distribution.mean} +/- {recall_distribution.standard_deviation}\n"
          f"\t* F1 Scores: {f1_distribution.mean} +/- {f1_distribution.standard_deviation}")