from sklearn.metrics import f1_score

def klue_re_micro_f1(preds, labels):
    return f1_score(labels, preds, average="micro") * 100.0