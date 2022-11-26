# import comet_ml
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
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
                            get_comet_experiment, \
                            scale_features

from milestone_2_question_6 import removeInvalidData

sns.set(style="darkgrid")


def main():

    # Add Experiments to Comet
    experiment = get_comet_experiment()
    experiment.set_name("question-3")
    experiment.add_tags(["question-3", "classifier", "base"])

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

    train, test = train_test_split(features, test_size=0.2, shuffle=True)
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
    for feats, model_name in zip(
        [
            ['Base Uniform Model'],
            ['shot_distance'],
            ['shot_angle'],
            ['shot_distance', 'shot_angle'],
            # ['hand_based_shot_angle'],
            # ['hand_based_shot_angle', 'shot_distance'],
        ],
        [
            '',
            'shot_distance_model.pkl',
            'shot_angle_model.pkl',
            'shot_distance_shot_angle_model.pkl',
        ]
    ):
        if feats[0] == 'Base Uniform Model':
            y_probas = [0.5] * len(test)
        else:
            # X_train = train[feats]
            # if len(feats) == 1:
            #     X_train = np.array(train[feats]).reshape(-1, 1)
            #     X_val = np.array(val[feats]).reshape(-1, 1)
            # else:
            #     pass
            X_train = train[feats].values
            clf.fit(X_train, y_train)
            pickle.dump(clf, open(model_name, 'wb'))

            X_test = test[feats].values
            # X_test = np.array(val[feats]).reshape(-1, 1)
            y_test = np.array(test['is_goal']).reshape(-1, 1)
            y_test = [y[0] for y in y_test]
            y_pred = clf.predict(X_test)
            print('Predictions Mean:', np.mean(y_pred), '\n')

            acc = accuracy_score(y_test, y_pred)
            print('Accuracy:', acc, '\n')
            # Questions 3.2 et 3.3
            y_probas = clf.predict_proba(X_test)
            y_probas = [y[1] for y in y_probas]

            experiment.log_parameters(clf.get_params)

        models_probas.append(y_probas)

    # a)
    for y_probas, label in zip(models_probas, labels):
        # print(y_probas)
        fpr, tpr, thresholds = roc_curve(y_test, y_probas)
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
    plt.savefig('./plots/question_3/ROC-question_3.jpeg')
    plt.show()

    # b)
    plot_goals_rate_pdf(y_test, models_probas, labels, image_file_name='./plots/question_3/goals_pdf_base_models.jpeg')
    plt.show()

    # c)
    plot_goals_rate_cdf(y_test, models_probas, labels, image_file_name='./plots/question_3/goals_cdf_base_models.jpeg')
    plt.show()

    # d)
    CalibrationDisplay.from_predictions(y_test, y_probas)
    plt.savefig('./plots/question_3/calibration-question_3.jpeg')
    plt.show()

    experiment.log_dataframe_profile(
        X_train,
        name='X_train',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )
    experiment.log_dataframe_profile(
        X_test,
        name='X_test',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )
    experiment.log_dataframe_profile(
        y_train,
        name='y_train',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )
    experiment.log_dataframe_profile(
        y_test,
        name='y_test',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )
    experiment.log_image('./plots/question_3/ROC-question_3.jpeg')
    experiment.log_image('./plots/question_3/goals_pdf_base_models.jpeg')
    experiment.log_image('./plots/question_3/goals_cdf_base_models.jpeg')
    experiment.log_image('./plots/question_3/calibration-question_3.jpeg')

    experiment.log_model("Linear Regressor (shot_distance)", "shot_distance_model.pkl")
    experiment.log_model("Linear Regressor (shot_angle)", "shot_angle_model.pkl")
    experiment.log_model("Linear Regressor (shot_distance & shot_angle)", "shot_distance_shot_angle_model.pkl")


if __name__ == "__main__":
    main()
