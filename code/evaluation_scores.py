from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np

### metrics' calculation 

def calculate_performace(num, y_pred, y_prob, y_test):
    tp, fp, tn, fn = 0, 0, 0, 0
    for index in range(num):
        if y_test[index] ==1:
            if y_test[index] == y_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if y_test[index] == y_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    acc = float(tp + tn)/num
    try:
        precision = float(tp)/(tp + fp)
        recall = float(tp)/ (tp + fn)
        f1_score = float((2*precision*recall)/(precision+recall))
        MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        sens = float(tp)/(tp+fn)
        spec = float(tn)/(tn+fp)
    except ZeroDivisionError:
        print("You can't divide by 0.")
        precision=recall=f1_score =sens = MCC=100
    AUC = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test,y_prob)

    return tp, fp, tn, fn, acc, precision, sens, f1_score, MCC, AUC,auprc,spec

