import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from constant import GG_SUM_TO_LABEL
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import roc_auc_score

def assign_image_primary_secondary(primary_classes, secondary_classes):
    # Sanity conditions
    for i, (primary, secondary) in enumerate(zip(primary_classes, secondary_classes)):
        if (primary > 0) and (secondary == 0):
            secondary = primary
        if (secondary > 0) and (primary == 0):
            primary = secondary

        primary_classes[i] = primary
        secondary_classes[i] = secondary
    return primary_classes, secondary_classes


def compute_metrics(targets, logits, task_names, is_gleason, **kwargs):
    num_tasks = len(task_names.split(','))

    # grouping and majority voting
    all_predictions = []
    all_targets = []
    all_probs = []
    for i in range(num_tasks):
        predictions_ = [x[i].detach().cpu().numpy() for x in logits]
        targets_ = [x[i].detach().cpu().numpy() for x in targets]

        predictions_ = np.vstack(predictions_)
        targets_ = np.vstack(targets_)
        probs_ = F.softmax(torch.from_numpy(predictions_), dim=1)

        all_predictions.append(np.vstack(predictions_))
        all_targets.append(np.vstack(targets_))
        all_probs.append(probs_)

    all_predictions = [np.argmax(x, axis=1) for x in all_predictions]
    all_targets = [np.argmax(x, axis=1) for x in all_targets]

    # cT_STAGE, CLINICAL_PROG, DISEASE_PROG, OS_STATUS
    if num_tasks == 1:
        targets = all_targets[0]
        predictions = all_predictions[0]
        weighted_f1 = f1_score(targets, predictions, average='weighted')

    # (cGS_PAT_1, cGS_PAT_2)
    elif num_tasks == 2 and is_gleason:
        primary_preds, secondary_preds = assign_image_primary_secondary(all_predictions[0], all_predictions[1])
        primary_targets, secondary_targets = all_targets

        # Gleason weighted F1 calculation
        predictions = []
        for key in (primary_preds + secondary_preds):
            predictions.append(GG_SUM_TO_LABEL[key])

        targets = []
        for key in (primary_targets + secondary_targets):
            targets.append(GG_SUM_TO_LABEL[key])

        weighted_f1 = f1_score(targets, predictions, average='weighted')

    else:
        print('ERROR: multi-task learning is currently not supported')
        exit()

    return weighted_f1
