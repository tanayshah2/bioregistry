import pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

file_path = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTTXz-iQf5zlfT3j0i7IpyfTMDUmvjb8Imz0bypGoxmcHwhr-ciwSc2a6nc2q-ldFMApgA4TUypJ0Z4/pub?gid=2110949093&single=true&output=csv'
df = pd.read_csv(file_path)

df['label'] = df['relevance'].apply(lambda x: 0 if x == 0 else 1)
df = df.dropna(subset=['meta_score'])  # Specify the subset parameter correctly

def analyze_and_plot(scores, label, title_suffix):
    scores_relevant = scores[df['label'] == 1]
    scores_irrelevant = scores[df['label'] == 0]

    t_stat, p_value = ttest_ind(scores_relevant, scores_irrelevant)

    precision, recall, _ = precision_recall_curve(df['label'], scores)
    auc_roc = roc_auc_score(df['label'], scores)
    fpr, tpr, _ = roc_curve(df['label'], scores)

    precision_smooth = np.interp(np.linspace(0, 1, 100), recall[::-1], precision[::-1])
    recall_smooth = np.linspace(0, 1, 100)
    fpr_smooth = np.interp(np.linspace(0, 1, 100), fpr, fpr)
    tpr_smooth = np.interp(np.linspace(0, 1, 100), fpr, tpr)

    print(f"t-test results for {title_suffix}:")
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_value}")
    print(f"\nAUC-ROC for {title_suffix}: {auc_roc}")

    plt.figure()
    plt.plot(recall_smooth, precision_smooth, label=f'Precision-Recall curve ({title_suffix})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({title_suffix})')
    plt.legend(loc="lower left")
    plt.show()

    plt.figure()
    plt.plot(fpr_smooth, tpr_smooth, label=f'ROC curve (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve ({title_suffix})')
    plt.legend(loc="lower right")
    plt.show()

analyze_and_plot(df['meta_score'], df['label'], 'Meta Classifier')

