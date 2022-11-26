import pandas as pd
import numpy as np
import generic_util
from comet_ml import Experiment
import os


def add_features_sub_question_1(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get features: period, coordinates, shot type, period seconds, shot distance and angle
    """
    
    start_time = pd.to_datetime(raw_df['game_start_time'], infer_datetime_format=True)
    end_time = pd.to_datetime(raw_df['game_end_time'], infer_datetime_format=True)
    raw_df['game_period_seconds'] = (end_time - start_time).dt.seconds
    raw_df['shot_distance'] = raw_df.apply(lambda x: generic_util.get_shot_distance(x.coordinates_x, x.coordinates_y, x.team_rink_side_right), axis=1)
    
    raw_df['rigth_side_ref_x'] = np.where(raw_df['team_rink_side_right'], -raw_df['coordinates_x'], raw_df['coordinates_x'])
    raw_df['rigth_side_ref_y'] = np.where(raw_df['team_rink_side_right'], -raw_df['coordinates_y'], raw_df['coordinates_y'])
    raw_df['shot_angle'] = raw_df.apply(lambda x: generic_util.get_signed_shot_angle(x.rigth_side_ref_x, x.rigth_side_ref_y), axis=1)

    raw_df['is_goal'] = np.where(raw_df['event_type'] == 'GOAL', 1, 0)
    raw_df['empty_net'] = np.where(raw_df['goal_empty_net'].isna(), 0, np.where(raw_df['goal_empty_net'], 1, 0))

    return raw_df


def add_features_sub_question_2(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    previous_event_type = None
    previous_event_coord_x = 0
    previous_event_coord_y = 0
    previous_event_dateTime = None
    for ind in raw_df.index:
        if raw_df['event_type'][ind] == 'SHOT' or raw_df['event_type'][ind] == 'GOAL':
            filter = (clean_df['event_id'] == raw_df['event_id'][ind]) & (clean_df['game_id'] == raw_df['game_id'][ind])

            row_index = clean_df[filter].index[0]
    
            clean_df.loc[row_index, 'hand_based_shot_angle'] = generic_util.get_shot_angle(raw_df['coordinates_x'][ind], raw_df['coordinates_y'][ind], raw_df['team_rink_side_right'][ind], raw_df['shooter_right_handed'][ind])

            clean_df.loc[row_index, 'last_event_type'] = previous_event_type
            
            clean_df.loc[row_index, 'last_coordinates_x'] = previous_event_coord_x
            clean_df.loc[row_index, 'last_coordinates_y'] = previous_event_coord_y

            event_time = pd.to_datetime(raw_df['dateTime'][ind])
            previous_event_time = pd.to_datetime(previous_event_dateTime)
            clean_df.loc[row_index, 'time_since_last_event'] = (event_time - previous_event_time).total_seconds() 

            game_start_time = pd.to_datetime(raw_df['game_start_time'][ind])
            clean_df.loc[row_index, 'game_elapsed_time'] = (event_time - game_start_time).total_seconds()

            if(previous_event_coord_x is None):
                print('None coordinate x found')
                previous_event_coord_x = 0

            if(previous_event_coord_y is None):
                print('None coordinate y found')
                previous_event_coord_y = 0

            clean_df.loc[row_index, 'distance_from_last_event'] = np.sqrt((raw_df['coordinates_x'][ind] - previous_event_coord_x)**2 + (raw_df['coordinates_y'][ind] - previous_event_coord_y)**2)

        previous_event_type = raw_df['event_type'][ind]
        previous_event_coord_x = raw_df['coordinates_x'][ind]
        previous_event_coord_y = raw_df['coordinates_y'][ind]
        previous_event_dateTime = raw_df['dateTime'][ind]

    return clean_df


def add_features_sub_question_3(clean_df: pd.DataFrame) -> pd.DataFrame:
    clean_df['rebond'] = clean_df['last_event_type'] == 'SHOT'
    clean_df['speed_from_last_event'] = clean_df['distance_from_last_event'] / clean_df['time_since_last_event']

    for ind in clean_df.index:
        if clean_df['rebond'][ind]:
            previous_shot_angle = clean_df['hand_based_shot_angle'][ind-1]
            current_shot_angle = clean_df['hand_based_shot_angle'][ind]
            if previous_shot_angle > 90:
                previous_shot_angle = 180 - previous_shot_angle
            if current_shot_angle > 90:
                current_shot_angle = 180 - current_shot_angle

            angle_change = 0 

            if clean_df['coordinates_y'][ind] * clean_df['last_coordinates_y'][ind] > 0:
                angle_change = np.abs(current_shot_angle - previous_shot_angle)
            else:
                angle_change = np.abs(180 - (current_shot_angle + previous_shot_angle))

            clean_df.loc[ind, 'shot_angle_change'] = angle_change

    return clean_df


def upload_exemple_sub_question_5(df: pd.DataFrame):
    
    game_id = 2017021065

    subset_df = df[df["game_id"] == game_id]
    subset_df = subset_df[['event_id', 'game_id', 'period', 'coordinates_x', 'coordinates_y', 'shot_type', 'game_period_seconds', 'game_elapsed_time', 'shot_distance', 'shot_angle', 'hand_based_shot_angle', 'is_goal', 'empty_net', 'last_event_type', 'last_coordinates_x', 'last_coordinates_y', 'time_since_last_event', 'distance_from_last_event', 'rebond', 'speed_from_last_event', 'shot_angle_change']]

    experiment = generic_util.get_comet_experiment()

    experiment.log_dataframe_profile(
        subset_df, 
        name='wpg_v_wsh_2017021065',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )

    print("Data uploaded to comet!")



if __name__ == "__main__":
    csv_path = './data/train.csv'
    df = pd.read_csv(csv_path)

    df = add_features_sub_question_1(df)
    df.to_csv('./data/train-q4-1.csv', index=False)

    raw_csv_path = './data/train-raw.csv'
    raw_df = pd.read_csv(raw_csv_path)
    df = add_features_sub_question_2(raw_df, df)
    df.to_csv('./data/train-q4-2.csv', index=False)

    df = add_features_sub_question_3(df)
    df.to_csv('./data/train-q4-3.csv', index=False)

    # csv_path = './data/train-q4-3.csv'
    # df = pd.read_csv(csv_path)
    upload_exemple_sub_question_5(df)



