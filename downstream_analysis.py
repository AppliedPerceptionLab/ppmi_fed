import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from utils import update_downstream_clf


def learn_classification(ppmi, config):
    for i in ['1', '2', '3']:
        ppmi[f'updrs{i}_status'] = ppmi[f'updrs{i}_score'] - ppmi.groupby('PATNO')[f'updrs{i}_score'].transform('first')
        ppmi[f'updrs{i}_status'] = ppmi[f'updrs{i}_status'].apply(lambda x: 0 if x <= 0 else 1)

    ppmi[f'updrstot_status'] = ppmi[f'updrs_totscore'] - ppmi.groupby('PATNO')[f'updrs_totscore'].transform('first')
    ppmi[f'updrstot_status'] = ppmi[f'updrstot_status'].apply(lambda x: 0 if x <= 0 else 1)

    drop_cols = ['PATNO', 'updrs1_score', 'updrs2_score', 'updrs3_score', 'updrs_totscore',
                 'updrs1_status', 'updrs2_status', 'updrs3_status', 'updrstot_status']
    if config['downstream_column'] == 'updrs1_score':
        test_col = 'updrs1_status'
    if config['downstream_column'] == 'updrs2_score':
        test_col = 'updrs2_status'
    if config['downstream_column'] == 'updrs3_score':
        test_col = 'updrs3_status'
    if config['downstream_column'] == 'updrs_totscore':
        test_col = 'updrstot_status'
    test_cols = [test_col]

    X_1 = ppmi.drop(drop_cols, axis=1)
    y = ppmi[test_cols]

    k_best = SelectKBest(f_classif, k=50)
    X = k_best.fit_transform(X_1, y)
    # X = X_1
    # print("Selected columns:", X_1.columns[k_best.get_support()])

    clf = xgb.XGBClassifier()

    tprs = []
    aucs = []
    accs = []
    f1_scores = []
    mean_fpr = np.linspace(0, 1, 100)
    pr_aucs = []
    y_real = []
    y_proba = []

    cv = StratifiedKFold(n_splits=5)
    for fold, (train, test) in enumerate(cv.split(X, y)):
        clf.fit(X[train], y.iloc[train].values.ravel())
        y_pred = clf.predict(X[test])

        accs.append(accuracy_score(y.iloc[test].values.ravel(), y_pred))
        f1_scores.append(f1_score(y.iloc[test].values.ravel(), y_pred))

        y_pred_proba = clf.predict_proba(X[test])[::, 1]
        aucs.append(roc_auc_score(y.iloc[test].values.ravel(), y_pred_proba))

        precision, recall, _ = precision_recall_curve(y.iloc[test].values.ravel(), y_pred_proba)
        pr_aucs.append(auc(recall, precision))

        y_real.append(y.iloc[test].values.ravel())
        y_proba.append(y_pred_proba)

        fpr, tpr, _ = roc_curve(y.iloc[test].values.ravel(), y_pred_proba)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    update_downstream_clf(config, mean_fpr, mean_tpr, auc(mean_fpr, mean_tpr), np.std(aucs), pr_aucs, np.mean(pr_aucs, axis=0), np.std(pr_aucs),
                          precision, recall)

    return np.mean(accs, axis=0), np.std(accs), np.mean(f1_scores, axis=0), np.std(f1_scores)
