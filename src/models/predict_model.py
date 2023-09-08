import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score,precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from hyperopt import fmin, tpe, hp, Trials

class Modeling:
    def __init__(self, model, X_train, X_val, y_train, y_val, X_test):
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.x_test = X_test

    def predict_model(self):
        y_train_preds = self.model.predict(self.X_train)
        y_val_preds = self.model.predict(self.X_val)

        y_train_probs = self.model.predict_proba(self.X_train)[:, 1]
        y_val_probs = self.model.predict_proba(self.X_val)[:, 1]

        train_score = pd.DataFrame({'roc-auc score': roc_auc_score(self.y_train, y_train_probs),
                                    'accuracy score': accuracy_score(self.y_train, y_train_preds),
                                    'precision score': precision_score(self.y_train, y_train_preds),
                                    'recall score': recall_score(self.y_train, y_train_preds),
                                    'f1 score': f1_score(self.y_train, y_train_preds)}, index=['Training Set'])

        val_score = pd.DataFrame({'roc-auc score': roc_auc_score(self.y_val, y_val_probs),
                                  'accuracy score': accuracy_score(self.y_val, y_val_preds),
                                  'precision score': precision_score(self.y_val, y_val_preds),
                                  'recall score': recall_score(self.y_val, y_val_preds),
                                  'f1 score': f1_score(self.y_val, y_val_preds)}, index=['Validation Set'])

        score = pd.concat([train_score, val_score])
        print(score)

        fpr_train, tpr_train, _ = roc_curve(self.y_train, y_train_probs)
        fpr_val, tpr_val, _ = roc_curve(self.y_val, y_val_probs)

        plt.plot(fpr_train, tpr_train, label='Train ROC-curve')
        plt.plot(fpr_val, tpr_val, label='Validation ROC-curve')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-Curve')
        plt.legend()

        ConfusionMatrixDisplay.from_estimator(self.model, self.X_train, self.y_train, normalize='true')
        ConfusionMatrixDisplay.from_estimator(self.model, self.X_val, self.y_val, normalize='true')

        plt.show()
        
    def hyperopt(self, parameters, max_evals=50):
        def objective(params):
            model = self.model.set_params(**params)
            
            X_train_val = pd.concat([pd.DataFrame(self.X_train), pd.DataFrame(self.X_val)])
            y_train_val = pd.concat([pd.Series(self.y_train), pd.Series(self.y_val)])
    
            model.fit(X_train_val, y_train_val)

            y_val_probs = model.predict_proba(self.X_val)[:, 1]
            score = -roc_auc_score(self.y_val, y_val_probs)

            return score

        trials = Trials()

        best = fmin(fn=objective, space=parameters, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        best_params = best

        print("Best Hyperparameters:")
        print(best_params)

        return best_params

    def permutation_importance(self):
        result = permutation_importance(self.model, self.X_val, self.y_val, n_repeats=30, random_state=8)

        importances = result.importances_mean
        feature_names = self.X_val.columns
        
        permutation_score = pd.DataFrame({'Feature Names': feature_names,
                                          'Permutation Importances': importances})
        permutation_score.sort_values(by='Permutation Importances', ascending=False, inplace=True)
        
        plt.figure(figsize=(20, 10))
        sns.barplot(data=permutation_score, x='Feature Names', y='Permutation Importances', palette='ch:.25')
        plt.ylabel('Permutation Importances')
        plt.xlabel('Feature Names')
        plt.xticks(rotation=90)
        plt.title('Permutation Importance')
        
        plt.show()

        return permutation_score