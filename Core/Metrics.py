from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics

class Metrics:

    def __init__(self):
        return 

    def ClassificationMetrics(self, true_y, pred_y):
        metrics = {
            "F1": self.F1(true_y, pred_y),
            "Area Under Curve": self.AUC(true_y, pred_y),
            "Confusion Matrix": self.ConfusionMatrix(true_y, pred_y)
        }
        return metrics

    def F1(self, true_y, pred_y):
        return f1_score(true_y, pred_y)
    
    def AUC(self, true_y, pred_y):
        fpr, tpr, thresholds = metrics.roc_curve(true_y, pred_y)
        metrics.roc_curve(true_y, pred_y)
        return metrics.roc_auc_score(true_y, pred_y)
    
    def ConfusionMatrix(self, true_y, pred_y):
        return confusion_matrix(true_y, pred_y)