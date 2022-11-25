# import comet_ml
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibrationDisplay

from generic_util import plot_goals_rate_pdf, \
                            plot_goals_rate_cdf, \
                            scale_features

from milestone_2_question_6 import removeInvalidData

sns.set(style="darkgrid")


def main():

    # Question 3.1
    features = pd.read_csv('data/train-q4-3.csv')
    features = features[~features['shot_distance'].isna()]
    features = features[~features['shot_angle'].isna()]
    features = features[~features['is_goal'].isna()]
    print(features[['shot_distance', 'shot_angle', 'hand_based_shot_angle', 'is_goal']])

    # Remove invalid goal
    features = removeInvalidData(features)

    features['coordinates_x'] = np.where(features['team_rink_side_right'], -features['coordinates_x'], features['coordinates_x'])
    features['coordinates_y'] = np.where(features['team_rink_side_right'], -features['coordinates_y'], features['coordinates_y'])

    features = features[['shot_distance', 'shot_angle', 'hand_based_shot_angle', 'is_goal']]
    # Scale Features
    # features[['shot_distance', 'shot_angle', 'hand_based_shot_angle']] = scale_features(features[['shot_distance', 'shot_angle', 'hand_based_shot_angle']])

    features['shot_angle_side'] = pd.np.sign(features['shot_angle'])
    features['shot_angle'] = features['shot_angle'].abs()
    print(features[['shot_distance', 'shot_angle', 'shot_angle_side', 'is_goal']])

    print(features['is_goal'].value_counts())

    train, val = train_test_split(features, test_size=0.2, shuffle=True)
    y_train = np.array(train['is_goal']).reshape(-1, 1)
    clf = LogisticRegression()

    models_probas = []
    labels = [
        'Base Uniform Model',
        'shot_distance',
        'shot_angle',
        'shot_distance & shot_angle',
        # 'hand_based_shot_angle',
        # 'hand_based_shot_angle & shot_distance',
    ]
    for feats in [
        ['Base Uniform Model'],
        ['shot_distance'],
        ['shot_angle'],
        ['shot_distance', 'shot_angle'],
        # ['hand_based_shot_angle'],
        # ['hand_based_shot_angle', 'shot_distance'],
    ]:
        if feats[0] == 'Base Uniform Model':
            y_probas = [0.5] * len(val)
        else:
            # X_train = train[feats]
            # if len(feats) == 1:
            #     X_train = np.array(train[feats]).reshape(-1, 1)
            #     X_val = np.array(val[feats]).reshape(-1, 1)
            # else:
            #     pass
            X_train = train[feats].values
            clf.fit(X_train, y_train)

            X_val = val[feats].values
            # X_val = np.array(val[feats]).reshape(-1, 1)
            y_val = np.array(val['is_goal']).reshape(-1, 1)
            y_val = [y[0] for y in y_val]
            y_pred = clf.predict(X_val)
            print('Predictions Mean:', np.mean(y_pred), '\n')

            acc = accuracy_score(y_val, y_pred)
            print('Accuracy:', acc, '\n')
            # Questions 3.2 et 3.3
            y_probas = clf.predict_proba(X_val)
            y_probas = [y[1] for y in y_probas]

        models_probas.append(y_probas)

    # a)
    for y_probas, label in zip(models_probas, labels):
        # print(y_probas)
        fpr, tpr, thresholds = roc_curve(y_val, y_probas)
        roc_auc = auc(fpr, tpr)
        # display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
        #                                   estimator_name=label)
        # display.plot()
        print(label)
        print('AUC:', auc(fpr, tpr), '\n')
        plt.plot(fpr, tpr, label=label)

    plt.legend()
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # b)
    plot_goals_rate_pdf(y_val, models_probas, labels)

    # c)
    plot_goals_rate_cdf(y_val, models_probas, labels)

    # d)
    disp = CalibrationDisplay.from_estimator(clf, X_val, y_val)
    plt.show()
    disp = CalibrationDisplay.from_predictions(y_val, y_probas)
    plt.show()

if __name__ == "__main__":
    main()
