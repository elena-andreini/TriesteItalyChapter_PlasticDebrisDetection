import torch

def metrics(y_pred, y_true):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    assert y_pred.shape == y_true.shape
    assert torch.all((y_pred == 0) | (y_pred == 1))
    assert torch.all((y_true == 0) | (y_true == 1))

    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    eps = 1e-8
    acc = tn / (tn + fp + eps)

    pos_precision = tp / (tp + fp + eps)
    pos_recall = tp / (tp + fn + eps)
    pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall + eps)
    pos_iou = tp / (tp + fp + fn + eps)

    neg_precision = tn / (tn + fn + eps)
    neg_recall = tn / (tn + fp + eps)
    neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall + eps)
    neg_iou = tn / (tn + fp + fn + eps)

    confusion_matrix = torch.tensor([[tn, fp], [fn, tp]])

    return {
        'accuracy': acc,
        'plastic_debris': {
            'precision': pos_precision,
            'recall': pos_recall,
            'f1': pos_f1,
            'iou': pos_iou
        },
        'background': {
            'precision': neg_precision,
            'recall': neg_recall,
            'f1': neg_f1,
            'iou': neg_iou
        },
        'confusion_matrix': confusion_matrix
    }