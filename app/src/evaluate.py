import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report

def evaluate_recurrent_network(model, X_test, y_test, batch_size=32):
    """
    Evaluates a recurrent neural network model using common classification metrics.

    Args:
        model: Trained recurrent neural network (e.g., Keras model).
        X_test: Test features (numpy array or tensor).
        y_test: True labels (numpy array).
        batch_size: Batch size for prediction.

    Returns:
        metrics: Dictionary with evaluation results.
    """
    # Predict probabilities and classes
    y_pred_prob = model.predict(X_test, batch_size=batch_size)
    if y_pred_prob.shape[-1] == 1:
        # Binary classification
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    else:
        # Multiclass classification
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_test = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0)
    }

    # ROC AUC (only for binary or multilabel indicator)
    # ROC AUC is a valid metric for binary and multiclass classification
    # For multiclass, we use 'ovr' (one-vs-rest) strategy
    try:
        if y_pred_prob.shape[-1] == 1:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_prob)
        else:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    except Exception:
        metrics['roc_auc'] = None

    return metrics

model = load_model('logs/my_rnn_model.h5')
metrics = evaluate_recurrent_network(model, X_test, y_test)
with open("classification_report.log", "w") as f:
    f.write(metrics['classification_report'])
