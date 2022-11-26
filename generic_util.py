import numpy as np
import pandas as pd
from comet_ml import Experiment
import matplotlib.pyplot as plt
import os
import scipy
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import math

COMET_PROJECT_NAME = "ift6758-project"
COMET_WORKSPACE = "ift6758-22-milestone-2"

NET_ABSOLUTE_COORD_X = 89
NET_COORD_Y = 0

sns.set(style="darkgrid")


def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    goals = df[df['is_goal'] == 1]
    duplicate = df.loc[goals.index.repeat(5)]
    df = pd.concat([df, duplicate]).reset_index(drop=True)
    return df


def get_comet_experiment() -> Experiment:
    """
    Creates a comet experiment with the right project configuration. 
    It will get the api key from this environment variable: COMET_API_KEY
    """
    comet_api_key = os.environ.get('COMET_API_KEY')
    experiment = Experiment(
        api_key=comet_api_key,
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE,
    )
    return experiment


def get_shot_distance(x: int, y: int, team_rink_side_right: bool) -> float:
    """
    Calculate the shot distance to the net

    Arguments:
    - x: shooter x coordinate on ice
    - y: shooter y coordinate on ice
    - team_rink_side_right: indicates if the shhoter rink is right
    """

    net_x = NET_ABSOLUTE_COORD_X
    
    if(team_rink_side_right):
        net_x = -NET_ABSOLUTE_COORD_X
        
    return np.sqrt((x-net_x)**2 + (y-NET_COORD_Y)**2)


def get_signed_shot_angle(rigth_side_ref_x: int, right_side_ref_y: int) -> float:
    """
    Calculate the signed angle from shot and x-absciss

    Arguments:
    - rigth_side_ref_x: x coord normalized to right rink side
    - rigth_side_ref_y: y coord normalized to right rink side
    """
    return math.degrees(math.atan2(NET_COORD_Y - right_side_ref_y, NET_ABSOLUTE_COORD_X - rigth_side_ref_x))


def get_shot_angle(x: int, y: int, team_rink_side_right: bool, shooter_right_handed: bool) -> float:
    """
    Calculate the shot angle from the net line

    Arguments:
    - x: shooter x coordinate on ice
    - y: shooter y coordinate on ice
    - team_rink_side_right: indicates if the shhoter rink is right
    - shooter_right_handed: indicates if shooter is right handed
    """

    angle = 0

    if y == 0:
        angle = 90
        return angle

    if(team_rink_side_right):
        net_x = -NET_ABSOLUTE_COORD_X
        adjacent = np.abs(y)
        opposite = np.abs(net_x - x)
        tan = opposite/adjacent
        angle = np.arctan(tan)
        if (y < 0 and not shooter_right_handed) or (y > 0 and shooter_right_handed):
            angle = np.pi - angle
    else:
        net_x = NET_ABSOLUTE_COORD_X
        adjacent = np.abs(y)
        opposite = np.abs(net_x - x)
        tan = opposite/adjacent
        angle = np.arctan(tan)
        if (y < 0 and shooter_right_handed) or (y > 0 and not shooter_right_handed):
            angle = np.pi - angle

    return math.degrees(angle)


def scale_features(features, feature_range=(0, 1)):
    """
    Scale the features DF between a given range.

    Arguments:
        - features: DF of the features we want to scale
        - feature_range: (Min, Max) tuple of the range we want the features values to be in

    Return:
        - Return the scale features DF
    """
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(features)
    scaled = scaler.transform(features)
    features = pd.DataFrame(scaled, index=features.index, columns=features.columns)

    return features


def plot_goals_rate_pdf(y_true, models_probas, labels, image_file_name=None):
    """
    Plot the PDF of the goals rate in function of the model prediction scores

    Arguments:
    - y_true: true values of the target
    - y_probas: lists of model probabilities predicted for a list of models
    """

    colors = ['b', 'r', 'g', 'orange']
    for y_probas, label, color in zip(models_probas, labels, colors):
        pdf = []
        x = []
        for i in range(100):
            threshold = np.percentile(y_probas, i)
            goals = len([y_prob for y_prob, y in zip(y_probas, y_true) if y_prob >= threshold and y == 1])
            non_goals = len([y_prob for y_prob, y in zip(y_probas, y_true) if y_prob >= threshold and y == 0])
            pdf.append(100*(goals / (goals + non_goals)))
            x.append(i)
        # sns.lineplot(x=x, y=pdf, label=label)
        plt.plot(x, pdf, color=color, label=label)

    plt.ylim(0, 100)
    plt.legend()
    plt.title('Goal Rate')
    plt.ylabel('Goals / (Goals + Shoots)')
    plt.xlabel('Shot Probability Model Percentile')
    if(image_file_name is None):
        plt.show()
    else:
        plt.savefig(image_file_name)



def plot_goals_rate_cdf(y_true, models_probas, labels, image_file_name=None):
    """
    Plot the PDF of the goals rate in function of the model prediction scores

    Arguments:
    - y_probas: model probabilities predicted
    - y_true: true values of the target
    """

    goals_tot = sum(y_true)
    colors = ['b', 'r', 'g', 'orange']
    for y_probas, label, color in zip(models_probas, labels, colors):
        cdf = []
        x = []
        # data = pd.DataFrame()
        for i in range(100):
            threshold = np.percentile(y_probas, i)
            goals = len([y_prob for y_prob, y in zip(y_probas, y_true) if y_prob >= threshold and y == 1])
            # goals = len([y_prob for y_prob, y in zip(y_probas, y_true) if y_prob <= threshold and y == 1])
            cdf.append(100*(goals / goals_tot))
            x.append(i)
        # data['x'] = x
        # data['cdf'] = cdf
        # sns.lineplot(data=data, x=x, y=cdf)
        plt.plot(x, cdf, color=color, label=label)

    plt.ylim(0, 100)
    plt.legend()
    plt.title('Cumulative % of goals')
    plt.ylabel('Proportion (%)')
    plt.xlabel('Shot Probability Model Percentile')
    if(image_file_name is None):
        plt.show()
    else:
        plt.savefig(image_file_name)


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data between train and test.
    Returns X_train, X_test, y_train, y_test

    Arguments:
    - df: the dataframe to split
    """
    train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=60)

    train = augment_data(train)

    y_train = np.array(train['is_goal']).reshape(-1, 1)
    y_train = [y[0] for y in y_train]

    y_test = np.array(test['is_goal']).reshape(-1, 1)
    y_test = [y[0] for y in y_test]

    X_train = train.drop('is_goal', axis=1)
    X_test = test.drop('is_goal', axis=1)
    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    print(np.pi/4)
    test = get_shot_distance(0, 0, True)
    test = get_shot_distance(-89, 10, True)

    test = get_shot_angle(0, 0, True, True)

    test = get_shot_angle(79, 10, False, True)
    test = get_shot_angle(79, 10, False, False)
    test = get_shot_angle(79, -10, False, True)
    test = get_shot_angle(79, -10, False, False)


    test = get_shot_angle(-79, 10, True, True)
    test = get_shot_angle(-79, 10, True, False)
    test = get_shot_angle(-79, -10, True, True)
    test = get_shot_angle(-79, -10, True, False)


    csv_path = './data/train.csv'

    df = pd.read_csv(csv_path)
    df['shot_angle'] = df.apply(lambda x: get_shot_angle(x.coordinates_x, x.coordinates_y, x.team_rink_side_right, x.shooter_right_handed), axis=1)
    df
