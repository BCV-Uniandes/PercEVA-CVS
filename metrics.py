# Adapted, effectively unmodified, from SwinCVS
# (https://github.com/franeknowak/SwinCVS/blob/main/scripts/f_metrics.py),
# licensed CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/).
# Credit: Nowak et al., "SwinCVS: a unified approach to classifying critical
# view of safety structures in laparoscopic cholecystectomy" (IJCARS, 2025).

# Standard library imports
from copy import deepcopy

# Third-party imports
import numpy as np
from sklearn.metrics import average_precision_score



# FUNCTIONS


def get_balanced_accuracies(y_true, y_pred):
    true_labels = np.concatenate([np.array(x) for x in y_true])
    predicted_labels = np.concatenate([np.array(x) for x in y_pred])

    C1_true = deepcopy(true_labels)
    C1_predicted = deepcopy(predicted_labels)
    C1_true = np.delete(C1_true, [1,2], axis=1)
    C1_predicted = np.delete(C1_predicted, [1,2], axis=1)

    C2_true = deepcopy(true_labels)
    C2_true = np.delete(C2_true, [0,2], axis=1)
    C2_predicted = deepcopy(predicted_labels)
    C2_predicted = np.delete(C2_predicted, [0,2], axis=1)

    C3_true = deepcopy(true_labels)
    C3_predicted = deepcopy(predicted_labels)
    C3_true = np.delete(C3_true, [0, 1], axis=1)
    C3_predicted = np.delete(C3_predicted, [0,1], axis=1)

    C1_recall = get_recall(C1_true, C1_predicted)
    C2_recall = get_recall(C2_true, C2_predicted)
    C3_recall = get_recall(C3_true, C3_predicted)

    C1_specificity = get_specificity(C1_true, C1_predicted)
    C2_specificity = get_specificity(C2_true, C2_predicted)
    C3_specificity = get_specificity(C3_true, C3_predicted)

    C1_balanced_accuracy = (C1_recall+C1_specificity)/2
    C2_balanced_accuracy = (C2_recall+C2_specificity)/2
    C3_balanced_accuracy = (C3_recall+C3_specificity)/2
    total_balanced_accuracy = (C1_recall+C2_recall+C3_recall)/3

    return C1_balanced_accuracy, C2_balanced_accuracy, C3_balanced_accuracy, total_balanced_accuracy

def get_recall(true, predicted):
    TP = np.sum((true == 1) & (predicted == 1))
    FN = np.sum((true == 1) & (predicted == 0))
    return TP / (TP + FN)


def get_specificity(true, predicted):
    TN = np.sum((true == 0) & (predicted == 0))
    FP = np.sum((true == 0) & (predicted == 1))
    return TN / (TN + FP)


def get_map(y_true, y_pred_probs_list):
    average_precisions = []

    true_labels = np.concatenate([np.array(x) for x in y_true])
    predicted_probabilities = np.concatenate([np.array(x) for x in y_pred_probs_list])
    for class_idx in range(true_labels.shape[1]):
        class_true = true_labels[:, class_idx]
        class_scores = predicted_probabilities[:, class_idx]
        average_precision = average_precision_score(class_true, class_scores)
        average_precisions.append(average_precision)

    # Calculate the mean of the average precisions across all classes to obtain mAP
    mAP = np.mean(average_precisions)
    C1_ap = average_precisions[0]
    C2_ap = average_precisions[1]
    C3_ap = average_precisions[2]
    
    return C1_ap, C2_ap, C3_ap, mAP

def find_best_epoch(results_dict):
    best_result = 0
    for key, val in results_dict.items():
        map_score = results_dict[key]['avg_map']
        if map_score >= best_result:
            best_epoch = key
            best_result = map_score
    return best_epoch
