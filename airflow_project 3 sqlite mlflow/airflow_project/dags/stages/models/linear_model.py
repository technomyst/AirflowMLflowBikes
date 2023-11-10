from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score


class LinearModel:

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X, need_proba=False):
        if need_proba:
            return self.model.predict_proba(X)[:, 1]
        if need_proba:
            return self.model.predict(X)

    def get_metrics(self, X_test, Y_test):
        predict = self.model.predict(X_test)
        predict_proba = self.model.predict_proba(X_test)[:,1]

        d = {"F1_score": f1_score(Y_test, predict),
             "Recall": recall_score(Y_test, predict),
             "Precision": precision_score(Y_test, predict),
             "ROC_AUC": roc_auc_score(Y_test, predict_proba)}

        return d



