import json
import requests
import pandas as pd
import logging
from os.path import exists
from os.path import join

from ift6758.client.serving_client import ServingClient
import nhl_tidy_data
import milestone_2_question_4
import milestone_2_question_6

NHL_API_DOMAIN = "https://statsapi.web.nhl.com"
LIVE_DATA_ENDPOINT = "/api/v1/game/{game_id}/feed/live/"

LAST_EVENTS_FILE_NAME = 'games_last_event.json'

logger = logging.getLogger(__name__)

class GameClient:
    """
    Class that returns predictions for games. It is a wrapper around the prediction api and the nhl api to predict for live events
    """
    def __init__(self, cache_folder_path: str = "./", ip = "127.0.0.1", port = 5000):
        self.cache_folder_path = cache_folder_path
        self.features = ['coordinates_x', 'coordinates_y', 'period', 'game_period_seconds', 'game_elapsed_time', 'shot_distance', 'shot_angle', 'hand_based_shot_angle', 'empty_net', 'last_coordinates_x', 'last_coordinates_y', 'time_since_last_event', 'distance_from_last_event', 'rebond', 'speed_from_last_event', 'shot_angle_change', 'ShotType_Backhand', 'ShotType_Deflected', 'ShotType_Slap Shot', 'ShotType_Snap Shot', 'ShotType_Tip-In', 'ShotType_Wrap-around', 'ShotType_Wrist Shot']
        self.serving_client = ServingClient(ip, port, self.features)


    def ping(self, game_id: str) -> pd.DataFrame:
        """
        Get event features and predictions for a specific `game_id`

        Args:
        - `game_id`: The game id

        Returns:
        - A pandas dataframe with all the features and the predictions
        """
        raw_game_data = self.download_game_data(game_id)

        games_last_event_file_path = join(self.cache_folder_path, f"{LAST_EVENTS_FILE_NAME}")
        games_last_event = {}

        if (exists(games_last_event_file_path)):
            with open(games_last_event_file_path) as f:
                games_last_event = json.load(f)
        
        last_event_id = None
        if(game_id in games_last_event):
            last_event_id = games_last_event[game_id]
        
        new_data_features, last_treated_event_id = self.get_new_data_features(raw_game_data, last_event_id, join(self.cache_folder_path, f"{game_id}-players"))

        game_predictions: pd.DataFrame = None
        game_predictions_file_path = join(self.cache_folder_path, f"{game_id}.csv")
        if (exists(game_predictions_file_path)):
            game_predictions = pd.read_csv(game_predictions_file_path)

        if new_data_features is not None:
            response = self.serving_client.predict(new_data_features)
            if(response['success']):
                new_data_features['goal_probability'] = response['data']['goal_probability'].values
                new_data_features['is_goal_prediction'] = response['data']['is_goal'].values
                if(game_predictions is None):
                    game_predictions = new_data_features
                else:
                    game_predictions = pd.concat([game_predictions, new_data_features])
            else:
                raise Exception(response['message'])

        if(game_predictions is not None):
            game_predictions.to_csv(game_predictions_file_path, index=False)
        
        games_last_event[game_id] = last_treated_event_id

        with open(games_last_event_file_path, 'w') as f:
            f.write(json.dumps(games_last_event))

        return game_predictions


    def get_new_data_features(self, game_data: dict, last_event_id: int, player_data_folder: str) -> tuple[pd.DataFrame, int]:
        """
        Get the data that was not yet processed
        """
        if len(game_data["liveData"]["plays"]["allPlays"]) == 0:
            return None, last_event_id
        
        result = []
        previous_event = None

        home_rink_side_right = nhl_tidy_data.get_period_home_rink_side_right(game_data["liveData"]["linescore"]["periods"])

        is_last_event_found = last_event_id is None

        for raw_event in game_data["liveData"]["plays"]["allPlays"]:
            event = nhl_tidy_data.get_event_features(game_data, raw_event, home_rink_side_right, player_data_folder)

            if(is_last_event_found):
                if event["event_type"] in ["SHOT", "GOAL"]:
                    if(previous_event is not None and previous_event["event_type"] not in ["SHOT", "GOAL"]):
                        result.append(previous_event)
                    result.append(event)
            else:
                is_last_event_found = event['event_id'] == last_event_id
            
            previous_event = event

        if(is_last_event_found):
            last_event_id = previous_event['event_id']

        response: pd.DataFrame = None

        if(len(result) > 0):
            raw_features = pd.DataFrame(result)
            
            features_of_interest = raw_features[(raw_features['event_type'] == "SHOT") | (raw_features['event_type'] == "GOAL")].copy()

            #Changing null data
            features_of_interest['team_rink_side_right'].fillna(False, inplace=True)
            features_of_interest['team_rink_side_right'] = features_of_interest['team_rink_side_right'].astype('bool')

            features_of_interest['coordinates_x'].fillna(0, inplace=True)
            features_of_interest['coordinates_y'].fillna(0, inplace=True)

            features_of_interest = milestone_2_question_4.add_features_sub_question_1(features_of_interest)

            raw_features['team_rink_side_right'].fillna(False, inplace=True)
            raw_features['team_rink_side_right'] = raw_features['team_rink_side_right'].astype('bool')

            raw_features['coordinates_x'].fillna(0, inplace=True)
            raw_features['coordinates_y'].fillna(0, inplace=True)
            
            features_of_interest = milestone_2_question_4.add_features_sub_question_2(raw_features, features_of_interest)

            #Changing null data
            features_of_interest['last_coordinates_x'].fillna(0, inplace=True)
            features_of_interest['last_coordinates_y'].fillna(0, inplace=True)
            
            response = milestone_2_question_4.add_features_sub_question_3(features_of_interest)

            #changing null data
            response['speed_from_last_event'].fillna(0, inplace=True)
            response = milestone_2_question_6.transform_data(response)

            for feature in self.features:
                if('ShotType_' in feature):
                    if(feature not in response.columns):
                        response[feature] = 0

        return response, last_event_id

    
    def download_game_data(self, game_id: str) -> dict:
        """
        Download game data from NHL api
        """
        game_data = None

        try:
            url = f"{NHL_API_DOMAIN}{LIVE_DATA_ENDPOINT.format(game_id = game_id)}"
            r = requests.get(url)
            if r.status_code == 200:
                game_data = r.json()
            else:
                message = f"Could not download game with id: {game_id} - status code: {r.status_code}"
                logger.info(message)
                raise Exception(message)
        except Exception as e:
            logger.info("====================================================================")
            message = f"Could not download game with id: {game_id}"
            logger.info(message)
            logger.info("====================================================================")
            raise Exception(message)

        return game_data


if __name__ == "__main__":
    game_client = GameClient("./data/predictions", "127.0.0.1", 9998)
    df = game_client.ping("2018021030")
    df.to_csv('./data/predictions/2018021030-pred.csv', index=False)