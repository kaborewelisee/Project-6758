import pandas as pd
import numpy as np
import generic_util
from comet_ml import Experiment

import math
import matplotlib.pyplot as plt
import numpy as np
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


def getRequiredFeatures(df: pd.DataFrame) -> pd.DataFrame:
    return df[['coordinates_x', 'coordinates_y', 'period', 'shot_type', 'game_period_seconds', 'shot_distance', 'shot_angle', 'hand_based_shot_angle', 'is_goal', 'empty_net', 'last_coordinates_x', 'last_coordinates_y', 'time_since_last_event', 'distance_from_last_event', 'rebond', 'speed_from_last_event', 'shot_angle_change']]


def removeInvalidData(df: pd.DataFrame) -> pd.DataFrame:
    #Remove missing team rink side
    df = df[~df['team_rink_side_right'].isnull()].copy()
    df['team_rink_side_right'] = df['team_rink_side_right'].astype('bool')
    #remove invalid last coordinate
    df = df[~(df['last_coordinates_x'].isnull() | df['last_coordinates_y'].isnull() | df['coordinates_x'].isnull() | df['coordinates_y'].isnull())]
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
    dummies_shot_type = pd.get_dummies(df.shot_type, prefix='ShotType')
    df = df.merge(dummies_shot_type, left_index=True, right_index=True)
    df.drop('shot_type', inplace=True, axis=1)
    return df


def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    goals = df[df['is_goal'] == 1]
    duplicate = df.loc[goals.index.repeat(5)]
    df = pd.concat([df, duplicate]).reset_index(drop=True)
    return df


def train_neural_net(x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.DataFrame, y_val):

    #x_train, x_val, y_train, y_val = generic_util.split_train_test(df)
    print()


if __name__ == "__main__":
    csv_path = './data/train-q4-3.csv'
    df = pd.read_csv(csv_path)

    df = removeInvalidData(df)
    df = getRequiredFeatures(df)
    df = transform_data(df)
    df = augment_data(df)

    x_train, x_val, y_train, y_val = generic_util.split_train_test(df)

    train_neural_net(x_train, x_val, y_train, y_val)


