import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibrationDisplay

from generic_util import plot_goals_rate_pdf, \
                            plot_goals_rate_cdf, \
                            get_comet_experiment


sns.set(style="darkgrid")


def main():

    # Add Experiments to Comet
    experiment = get_comet_experiment()
    experiment.set_name("question-3_test")
    experiment.add_tags(["question-3_test", "classifier", "base", "test"])

    # Question 7 partie 3
    test = pd.read_csv('data/test_remv_ReqF_Transform_aug.csv')

    models_probas = []
    labels = [
        'shot_distance',
        'shot_angle',
        'shot_distance & shot_angle',
    ]
    for feats, model_name in zip(
        [
            ['shot_distance'],
            ['shot_angle'],
            ['shot_distance', 'shot_angle'],
        ],
        [
            'shot_distance_model.pkl',
            'shot_angle_model.pkl',
            'shot_distance_shot_angle_model.pkl',
        ]
    ):
        with open(model_name, 'rb') as pickle_file:
            clf = pickle.load(pickle_file)

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
    plt.savefig('./plots/question_3_test/ROC-question_3.jpeg')
    plt.show()

    # b)
    plot_goals_rate_pdf(y_test, models_probas, labels, image_file_name='./plots/question_3_test/goals_pdf_base_models.jpeg')
    plt.show()

    # c)
    plot_goals_rate_cdf(y_test, models_probas, labels, image_file_name='./plots/question_3_test/goals_cdf_base_models.jpeg')
    plt.show()

    # d)
    CalibrationDisplay.from_predictions(y_test, y_probas)
    plt.savefig('./plots/question_3_test/calibration-question_3.jpeg')
    plt.show()

    experiment.log_image('./plots/question_3_test/ROC-question_3.jpeg')
    experiment.log_image('./plots/question_3_test/goals_pdf_base_models.jpeg')
    experiment.log_image('./plots/question_3_test/goals_cdf_base_models.jpeg')
    experiment.log_image('./plots/question_3_test/calibration-question_3.jpeg')


if __name__ == "__main__":
    main()
