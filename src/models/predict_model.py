import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def predict_model(model, X_train, X_val, y_train, y_val):
    y_train_probs = model.predict_proba(X_train)[:, 1]
    y_val_probs = model.predict_proba(X_val)[:, 1]

    train_score = pd.DataFrame({'roc-auc score': roc_auc_score(y_train, y_train_probs)}, index=['Training Set'])
    val_score = pd.DataFrame({'roc-auc score': roc_auc_score(y_val, y_val_probs)}, index=['Validation Set'])

    score = pd.concat([train_score, val_score])
    print(score)
    
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_probs)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_probs)
    
    plt.plot(fpr_train, tpr_train, label='Train ROC-curve')
    plt.plot(fpr_val, tpr_val, label='Validate ROC-curve')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-Curve')
    plt.legend()
    
    plt.show()