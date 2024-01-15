import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc

import numpy as np

import seaborn as sb
import matplotlib.pyplot as plt

"""
兵王问题
"""


def read_data():
    df = pd.read_csv(r'..\dataset\krkopt.data', header=None)
    df.dropna(inplace=True)
    return df


def normalize_data(df):
    for i in [0, 2, 4]:
        df.iloc[:, i] = df.iloc[:, i].map(lambda s: ord(s[0]) - 96)

    df.iloc[:, 6] = df.iloc[:, 6].map(lambda s: 1 if s == 'draw' else -1)

    # 训练和测试样本归一化
    for i in range(6):
        df[i] = (df[i] - df[i].mean()) / df[i].std()

    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :6], df[6], test_size=len(df) - 5000)
    return x_train, x_test, y_train, y_test


def f():
    df = read_data()
    x_train, x_test, y_train, y_test = normalize_data(df)

    # C 2**-5~2**15
    # gamma 2**-15~2**3
    CScale = [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
    gammaScale = [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3]
    cv_scores = 0
    c_max = None
    g_max = None
    for ci, c in enumerate(CScale):
        for gi, g in enumerate(gammaScale):
            model = SVC(kernel='rbf', C=2 ** c, gamma=2 ** g)
            # cross validation  5 cross validation,留一法
            # 5 cross validation
            scores = cross_val_score(model, x_train, y_train.astype('int'), cv=5,
                                     scoring='accuracy')  # y的标签需要是int或string类型
            if (m := scores.mean()) > cv_scores:
                cv_scores = m
                ci_max = ci
                c_max = c
                gi_max = gi
                g_max = g

    # 寻找更好的c和gamma
    n = 10
    min_c_scale = 0.5 * (CScale[max(0, ci_max - 1)] + CScale[ci_max])
    max_c_scale = 0.5 * (CScale[min(len(CScale) - 1, ci_max + 1)] + CScale[ci_max])
    new_c_scale_range = np.arange(min_c_scale, max_c_scale + 0.001, (max_c_scale - min_c_scale) / n)

    min_gamma_scale = 0.5 * (gammaScale[max(0, gi_max - 1)] + gammaScale[gi_max])
    max_gamma_scale = 0.5 * (gammaScale[min(len(gammaScale) - 1, gi_max + 1)] + gammaScale[gi_max])
    new_gamma_scale_range = np.arange(min_gamma_scale, max_gamma_scale + 0.001, (max_gamma_scale - min_gamma_scale) / n)

    for c in new_c_scale_range:
        for g in new_gamma_scale_range:
            model = SVC(kernel='rbf', C=2 ** c, gamma=2 ** g)
            scores = cross_val_score(model, x_train, y_train.astype('int'), cv=5,
                                     scoring='accuracy')  # y的标签需要是int或string类型
            if (m := scores.mean()) > cv_scores:
                cv_scores = m
                c_max = c
                g_max = g

    model = SVC(kernel='rbf', C=2 ** c_max, gamma=2 ** g_max)
    model.fit(x_train, y_train.astype('int'))
    pre = model.predict(x_test)
    model.score(x_test, y_test.astype('int'))

    cm = confusion_matrix(y_test.astype('int'), pre, labels=[1, -1], sample_weight=None)
    sb.set()
    f, ax = plt.subplots()
    sb.heatmap(cm, annot=True, ax=ax)  # 热力图
    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    fpr, tpr, threshold = roc_curve(y_test.astype('int'), pre)
    roc_auc = auc(fpr, tpr)  # 计算auc的值，auc就是曲线包含的面积，越大越好。 Compute Area Under the Curve (AUC) using the trapezoidal rule.

    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area={roc_auc:.2f}')
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example ')
    plt.legend(loc='lower right')
    plt.show()


def svm_c(x_train, x_test, y_train, y_test):
    svc = SVC(kernel='rbf', class_weight='balanced')
    c_range = np.logspace(-5, 15, 11, base=2)
    g_range = np.logspace(-9, 3, 13, base=2)
    param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': g_range}]
    grid = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1)

    clf = grid.fit(x_train, y_train.astype('int'))

    score = clf.score(x_test, y_test.astype('int'))
    print(f'精度为:{score}')
