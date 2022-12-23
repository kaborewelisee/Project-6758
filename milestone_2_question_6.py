import pandas as pd
import numpy as np
import generic_util

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibrationDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import time
import pickle


def getRequiredFeatures(df: pd.DataFrame) -> pd.DataFrame:
    return df[['coordinates_x', 'coordinates_y', 'period', 'shot_type', 'game_elapsed_time', 'shot_distance', 'shot_angle', 'hand_based_shot_angle', 'is_goal', 'empty_net', 'last_coordinates_x', 'last_coordinates_y', 'time_since_last_event', 'distance_from_last_event', 'rebond', 'speed_from_last_event', 'shot_angle_change']]


def removeInvalidData(df: pd.DataFrame) -> pd.DataFrame:
    #Remove missing team rink side
    df = df[~df['team_rink_side_right'].isnull()].copy()
    df['team_rink_side_right'] = df['team_rink_side_right'].astype('bool')
    #remove invalid coordinates
    df = df[~(df['last_coordinates_x'].isnull() | df['last_coordinates_y'].isnull() | df['coordinates_x'].isnull() | df['coordinates_y'].isnull())]
    #remove invalid speed_from_last_event
    df = df[~df['speed_from_last_event'].isnull()]
    #Remove invalid goal
    DEFENSIVE_ZONE_X = 25
    isNotEmptyNetGoal = (df['is_goal'] == 1) & (df['empty_net'] == 0)
    isFromRightDefense = (df['coordinates_x'] > DEFENSIVE_ZONE_X) & (df['team_rink_side_right'])
    isFromLeftDefense = (df['coordinates_x'] < -DEFENSIVE_ZONE_X) & (~df['team_rink_side_right'])
    isInvalidGoal = isNotEmptyNetGoal & (isFromRightDefense | isFromLeftDefense)
    df = df[~isInvalidGoal]
    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df['rebond'] = df['rebond'].astype(int)
    df['shot_angle_change'].fillna(0, inplace=True)
    #df['game_period_seconds'].fillna(df['game_period_seconds'].mean(), inplace=True)

    max_speed_from_last_event = df.loc[df['speed_from_last_event'] != np.inf, 'speed_from_last_event'].max()
    df['speed_from_last_event'].replace(np.inf, max_speed_from_last_event, inplace=True)

    dummies_shot_type = pd.get_dummies(df.shot_type, prefix='ShotType')
    df = df.merge(dummies_shot_type, left_index=True, right_index=True)
    df.drop('shot_type', inplace=True, axis=1)
    return df


def train_decision_tree(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test):

    experiment = generic_util.get_comet_experiment()

    experiment.set_name("question-6-decision-tree-classifer-base")
    experiment.add_tags(["question-6", "decistion-tree", "classifer", "base"])

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
        pd.DataFrame(y_train), 
        name='y_train',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )

    experiment.log_dataframe_profile(
        pd.DataFrame(y_test), 
        name='y_test',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    filename = './data/question-6-decision-tree-classifer/question-6-decision-tree-classifer-base.sav'
    pickle.dump(clf, open(filename, 'wb'))

    experiment.log_model('question-6-decision-tree-classifer-base', filename)

    models_probas = []
    labels = [
        'Base Uniform Model',
        'Decision Tree'
    ]

    y_probas = [0.5] * len(X_test)
    models_probas.append(y_probas)

    y_pred = clf.predict(X_test)
    print('Predictions Mean:', np.mean(y_pred), '\n')

    acc = accuracy_score(y_test, y_pred)
    print('Accuracy:', acc, '\n')

    y_probas = clf.predict_proba(X_test)
    y_probas = [y[1] for y in y_probas]
    models_probas.append(y_probas)


    #Create ROC plot
    for y_probas, label in zip(models_probas, labels):
        fpr, tpr, thresholds = roc_curve(y_test, y_probas)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=label)

    plt.legend()
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('./data/question-6-decision-tree-classifer/roc.jpeg')
    experiment.log_image('./data/question-6-decision-tree-classifer/roc.jpeg')

    #Create goals rate pdf plot
    generic_util.plot_goals_rate_pdf(y_test, models_probas, labels, './data/question-6-decision-tree-classifer/pdf.jpeg')
    experiment.log_image('./data/question-6-decision-tree-classifer/pdf.jpeg')

    #Create goals rate cdf plot
    generic_util.plot_goals_rate_cdf(y_test, models_probas, labels, './data/question-6-decision-tree-classifer/cdf.jpeg')
    experiment.log_image('./data/question-6-decision-tree-classifer/cdf.jpeg')

    #Create fiability curve from estimator
    disp = CalibrationDisplay.from_estimator(clf, X_test, y_test)
    plt.savefig('./data/question-6-decision-tree-classifer/calibration-estimator.jpeg')
    experiment.log_image('./data/question-6-decision-tree-classifer/calibration-estimator.jpeg')

    #Create fiability curve from prediction
    disp = CalibrationDisplay.from_predictions(y_test, y_probas)
    plt.savefig('./data/question-6-decision-tree-classifer/calibration-prediction.jpeg')
    experiment.log_image('./data/question-6-decision-tree-classifer/calibration-prediction.jpeg')

    time.sleep(3 * 60)

    print()


def train_random_forest(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test):

    experiment = generic_util.get_comet_experiment()

    experiment.set_name("question-6-random-forest-classifier-base")
    experiment.add_tags(["question-6", "random-forest", "classifier", "base"])

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
        pd.DataFrame(y_train), 
        name='y_train',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )

    experiment.log_dataframe_profile(
        pd.DataFrame(y_test), 
        name='y_test',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    filename = './data/question-6-random-forest-classifier/question-6-random-forest-base.sav'
    pickle.dump(clf, open(filename, 'wb'))

    experiment.log_model('question-6-random-forest-classifier-base', filename)

    models_probas = []
    labels = [
        'Base Uniform Model',
        'Random Forest'
    ]

    y_probas = [0.5] * len(X_test)
    models_probas.append(y_probas)

    y_pred = clf.predict(X_test)
    print('Predictions Mean:', np.mean(y_pred), '\n')

    acc = accuracy_score(y_test, y_pred)
    print('Accuracy:', acc, '\n')

    y_probas = clf.predict_proba(X_test)
    y_probas = [y[1] for y in y_probas]
    models_probas.append(y_probas)

    #Create ROC plot
    for y_probas, label in zip(models_probas, labels):
        fpr, tpr, thresholds = roc_curve(y_test, y_probas)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=label)

    plt.legend()
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('./data/question-6-random-forest-classifier/roc.jpeg')
    experiment.log_image('./data/question-6-random-forest-classifier/roc.jpeg')

    #Create goals rate pdf plot
    generic_util.plot_goals_rate_pdf(y_test, models_probas, labels, './data/question-6-random-forest-classifier/pdf.jpeg')
    experiment.log_image('./data/question-6-random-forest-classifier/pdf.jpeg')

    #Create goals rate cdf plot
    generic_util.plot_goals_rate_cdf(y_test, models_probas, labels, './data/question-6-random-forest-classifier/cdf.jpeg')
    experiment.log_image('./data/question-6-random-forest-classifier/cdf.jpeg')

    #create fiabily curve from estimator
    disp = CalibrationDisplay.from_estimator(clf, X_test, y_test)
    plt.savefig('./data/question-6-random-forest-classifier/calibration-estimator.jpeg')
    experiment.log_image('./data/question-6-random-forest-classifier/calibration-estimator.jpeg')

    #create fiabily curve from prediction
    disp = CalibrationDisplay.from_predictions(y_test, y_probas)
    plt.savefig('./data/question-6-random-forest-classifier/calibration-prediction.jpeg')
    experiment.log_image('./data/question-6-random-forest-classifier/calibration-prediction.jpeg')

    time.sleep(6 * 60)

    print()


def train_mlp_classifier(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test):

    experiment = generic_util.get_comet_experiment()

    experiment.set_name("question-6-mlp-classifier-base")
    experiment.add_tags(["question-6", "MLP", "classifier", "base"])

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
        pd.DataFrame(y_train), 
        name='y_train',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )

    experiment.log_dataframe_profile(
        pd.DataFrame(y_test), 
        name='y_test',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )

    max_iter = 50
    cv = 5
    n_jobs = -1
    nb_features = len(X_test.columns)
    mlp_grid_search = MLPClassifier(max_iter=max_iter, early_stopping=True)
    search_parameters = {
        'hidden_layer_sizes': [(nb_features, nb_features // 2, nb_features // 4), (nb_features,)],
        'activation': ['tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive']
    }

    clf = GridSearchCV(estimator=mlp_grid_search, param_grid=search_parameters, n_jobs=n_jobs, cv=cv)
    clf.fit(X_train.values, y_train)

    search_parameters['n_jobs'] = n_jobs
    search_parameters['cv'] = cv
    search_parameters['max_iter'] = max_iter
    experiment.log_parameters(search_parameters)
    experiment.log_parameters(clf.best_estimator_, prefix='best_param_')

    filename = './data/question-6-mlp-classifier/question-6-mlp-classifier-base.sav'
    pickle.dump(clf, open(filename, 'wb'))

    experiment.log_model('question-6-mlp-classifier-base', filename)

    models_probas = []
    labels = [
        'Base Uniform Model',
        'MLP'
    ]

    y_probas = [0.5] * len(X_test)
    models_probas.append(y_probas)

    y_pred = clf.predict(X_test.values)
    print('Predictions Mean:', np.mean(y_pred), '\n')

    acc = accuracy_score(y_test, y_pred)
    print('Accuracy:', acc, '\n')

    y_probas = clf.predict_proba(X_test.values)
    y_probas = [y[1] for y in y_probas]
    models_probas.append(y_probas)

    #Create ROC plot
    for y_probas, label in zip(models_probas, labels):
        fpr, tpr, thresholds = roc_curve(y_test, y_probas)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=label)

    plt.legend()
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('./data/question-6-mlp-classifier/roc.jpeg')
    experiment.log_image('./data/question-6-mlp-classifier/roc.jpeg')

    #Create goals rate pdf plot
    generic_util.plot_goals_rate_pdf(y_test, models_probas, labels, './data/question-6-mlp-classifier/pdf.jpeg')
    experiment.log_image('./data/question-6-mlp-classifier/pdf.jpeg')

    #Create goals rate cdf plot
    generic_util.plot_goals_rate_cdf(y_test, models_probas, labels, './data/question-6-mlp-classifier/cdf.jpeg')
    experiment.log_image('./data/question-6-mlp-classifier/cdf.jpeg')

    #create fiabily curve from estimator
    disp = CalibrationDisplay.from_estimator(clf, X_test.values, y_test)
    plt.savefig('./data/question-6-mlp-classifier/calibration-estimator.jpeg')
    experiment.log_image('./data/question-6-mlp-classifier/calibration-estimator.jpeg')

    #create fiabily curve from prediction
    disp = CalibrationDisplay.from_predictions(y_test, y_probas)
    plt.savefig('./data/question-6-mlp-classifier/calibration-prediction.jpeg')
    experiment.log_image('./data/question-6-mlp-classifier/calibration-prediction.jpeg')

    print()


def train_mlp_classifier_scaled(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test):

    experiment = generic_util.get_comet_experiment()

    experiment.set_name("question-6-mlp-classifier-scaled")
    experiment.add_tags(["question-6", "MLP", "classifier", "scaled"])

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
        pd.DataFrame(y_train), 
        name='y_train',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )

    experiment.log_dataframe_profile(
        pd.DataFrame(y_test), 
        name='y_test',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )

    nb_features = len(X_test.columns)

    scaler = StandardScaler()
    scaler = scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    max_iter = 50
    cv = 5
    n_jobs = -1
    mlp_grid_search = MLPClassifier(max_iter=max_iter, early_stopping=True)
    search_parameters = {
        'hidden_layer_sizes': [(nb_features, nb_features // 2, nb_features // 4), (nb_features,)],
        'activation': ['tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive']
    }

    clf = GridSearchCV(estimator=mlp_grid_search, param_grid=search_parameters, n_jobs=n_jobs, cv=cv)
    clf.fit(X_train, y_train)

    search_parameters['n_jobs'] = n_jobs
    search_parameters['cv'] = cv
    search_parameters['max_iter'] = max_iter
    experiment.log_parameters(search_parameters)
    # experiment.log_parameters(clf.best_estimator_, prefix='best_param_')

    filename = './data/question-6-mlp-classifier-scaled/question-6-mlp-classifier-scaled.sav'
    pickle.dump(clf, open(filename, 'wb'))

    experiment.log_model('question-6-mlp-classifier-scaled', filename)

    models_probas = []
    labels = [
        'Base Uniform Model',
        'MLP scaled'
    ]

    y_probas = [0.5] * len(X_test)
    models_probas.append(y_probas)

    y_pred = clf.predict(X_test)
    print('Predictions Mean:', np.mean(y_pred), '\n')

    acc = accuracy_score(y_test, y_pred)
    print('Accuracy:', acc, '\n')

    y_probas = clf.predict_proba(X_test)
    y_probas = [y[1] for y in y_probas]
    models_probas.append(y_probas)

    #Create ROC plot
    for y_probas, label in zip(models_probas, labels):
        fpr, tpr, thresholds = roc_curve(y_test, y_probas)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=label)

    plt.legend()
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('./data/question-6-mlp-classifier-scaled/roc.jpeg')
    experiment.log_image('./data/question-6-mlp-classifier-scaled/roc.jpeg')

    #Create goals rate pdf plot
    generic_util.plot_goals_rate_pdf(y_test, models_probas, labels, './data/question-6-mlp-classifier-scaled/pdf.jpeg')
    experiment.log_image('./data/question-6-mlp-classifier-scaled/pdf.jpeg')

    #Create goals rate cdf plot
    generic_util.plot_goals_rate_cdf(y_test, models_probas, labels, './data/question-6-mlp-classifier-scaled/cdf.jpeg')
    experiment.log_image('./data/question-6-mlp-classifier-scaled/cdf.jpeg')

    #create fiabily curve from estimator
    disp = CalibrationDisplay.from_estimator(clf, X_test, y_test)
    plt.savefig('./data/question-6-mlp-classifier-scaled/calibration-estimator.jpeg')
    experiment.log_image('./data/question-6-mlp-classifier-scaled/calibration-estimator.jpeg')

    #create fiabily curve from prediction
    disp = CalibrationDisplay.from_predictions(y_test, y_probas)
    plt.savefig('./data/question-6-mlp-classifier-scaled/calibration-prediction.jpeg')
    experiment.log_image('./data/question-6-mlp-classifier-scaled/calibration-prediction.jpeg')

    print()


def upload_to_comet(X_train, X_test, y_train, y_test):
    experiment = generic_util.get_comet_experiment()

    experiment.set_name("question-6-mlp-classifier-scaled")
    experiment.add_tags(["question-6", "MLP", "classifier", "scaled"])

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
        pd.DataFrame(y_train), 
        name='y_train',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )

    experiment.log_dataframe_profile(
        pd.DataFrame(y_test), 
        name='y_test',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )

    nb_features = len(X_test.columns)

    max_iter = 50
    cv = 5
    n_jobs = -1
    search_parameters = {
        'hidden_layer_sizes': [(nb_features, nb_features // 2, nb_features // 4), (nb_features,)],
        'activation': ['tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive']
    }

    search_parameters['n_jobs'] = n_jobs
    search_parameters['cv'] = cv
    search_parameters['max_iter'] = max_iter
    experiment.log_parameters(search_parameters)

    filename = './data/question-6-mlp-classifier-scaled/question-6-mlp-classifier-scaled.sav'
    experiment.log_model('question-6-mlp-classifier-scaled', filename)
    experiment.log_image('./data/question-6-mlp-classifier-scaled/roc.jpeg')
    experiment.log_image('./data/question-6-mlp-classifier-scaled/pdf.jpeg')
    experiment.log_image('./data/question-6-mlp-classifier-scaled/cdf.jpeg')
    experiment.log_image('./data/question-6-mlp-classifier-scaled/calibration-estimator.jpeg')
    experiment.log_image('./data/question-6-mlp-classifier-scaled/calibration-prediction.jpeg')

    time.sleep(4 * 60)
    print()






if __name__ == "__main__":
    csv_path = './data/train-q4-3.csv'
    df = pd.read_csv(csv_path)

    df = removeInvalidData(df)
    df = getRequiredFeatures(df)
    df = transform_data(df)


    X_train, X_test, y_train, y_test = generic_util.split_train_test(df)


    train_decision_tree(X_train, X_test, y_train, y_test)

    # train_random_forest(X_train, X_test, y_train, y_test)

    #train_mlp_classifier(X_train, X_test, y_train, y_test)

    #train_mlp_classifier_scaled(X_train, X_test, y_train, y_test)



