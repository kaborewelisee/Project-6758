# import comet_ml
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
sns.set(style="darkgrid")


def prob_plot(features, col):
    df_goal = features[features['is_goal'] == 1]
    df_not_goal = features[features['is_goal'] == 0]
    bins = np.linspace(min(features[col]), max(features[col]), 20)
    goal_hist = np.histogram(df_goal[col], bins=bins, range=(0, 80))
    not_goal_hist = np.histogram(df_not_goal[col], bins=bins, range=(0, 80))
    goal_rates = goal_hist[0] / (goal_hist[0] + not_goal_hist[0])
    plt.bar(bins[:-1], goal_rates, width=bins[1] - bins[0])
    plt.xlabel(col)
    plt.ylabel('Goal Rate')
    plt.title(f'Goal Probability per {col}')
    plt.show()


def makes_features():
    """
    Makes the basic features and save them to features/features_base.csv
    """

    # Question 2.1
    df = pd.read_csv('data/train.csv')
    df = df[~df['team_rink_side_right'].isnull()]
    # TODO filtrer training set matchs saison reguliere
    print(df['team_rink_side_right'].value_counts())
    print(df['team_rink_side_right'].describe())

    # TODO corriger queue buts distance 125-175
    df['coordinates_x'] = np.where(df['team_rink_side_right'], -df['coordinates_x'], df['coordinates_x'])
    df['coordinates_y'] = np.where(df['team_rink_side_right'], -df['coordinates_y'], df['coordinates_y'])

    net_pos = (89, 0)
    features = pd.DataFrame()
    features['dist_net'] = np.sqrt((net_pos[0] - df['coordinates_x'])**2 + (net_pos[1] - df['coordinates_y'])**2)
    features['angle_shoot'] = df.apply(
        lambda x: math.degrees(math.atan2(net_pos[1] - x['coordinates_y'], net_pos[0] - x['coordinates_x'])),
        axis=1
    )
    features['is_goal'] = np.where(df['event_type'] == 'GOAL', 1, 0)
    features['empty_net'] = np.where(df['goal_empty_net'].isna(), 0, np.where(df['goal_empty_net'], 1, 0))

    print(features[['is_goal', 'empty_net', 'dist_net', 'angle_shoot']])

    ax = sns.histplot(data=features, x="dist_net", hue="is_goal", multiple="stack").set(title='Shootings & Goals by Distance from Net')
    # ax.legend()
    # plt.legend()
    plt.show()

    sns.histplot(data=features, x="angle_shoot", hue="is_goal", multiple="stack").set(title='Shootings & Goals by Shooting Angle')
    # plt.legend()
    plt.show()

    sns.jointplot(data=features, x="dist_net", y="angle_shoot", hue="is_goal")
    # plt.legend()
    # plt.title('Shootings & Goals by Distance from Net & Shooting Angle')
    plt.show()

    # Question 2.2
    prob_plot(features, 'dist_net')
    prob_plot(features, 'angle_shoot')

    # Question 2.3
    df_goal = features[features['is_goal'] == 1]
    sns.histplot(data=df_goal, x="dist_net", hue="empty_net", multiple="stack").set(title='Goals by Distance from Net')
    # plt.legend()
    plt.show()
    sns.histplot(data=df_goal, x="angle_shoot", hue="empty_net", multiple="stack").set(title='Goals by Shooting Angle')
    # plt.legend()
    plt.show()

    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/features'):
        os.mkdir('data/features')
    features.to_csv('data/features/features_base.csv', index=False)


if __name__ == "__main__":
    makes_features()
