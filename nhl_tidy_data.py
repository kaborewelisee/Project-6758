from typing import List
import nhl_dataset
import glob
import os
import pandas as pd
import json
from os.path import exists
from os.path import join
import requests
from pathlib import Path

def convert_raw_data_to_panda_csv(nhl: nhl_dataset.NhlDataset, csv_path: str, seasons: List[int], game_types: List[str] = [nhl_dataset.REGULAR_GAME_TYPE, nhl_dataset.PLAYOFFS_GAME_TYPE]):
    """
    Converts the raw data from a nhl dataset to pandas dataframe and save it in a csv

    Arguments:
    - nhl: the NhlDataset object
    - csv_path: the path where to save the dataframe
    - seasons: the game seasons of interest
    - game_types: the type of game to include in the dataframe. SEE nhl_dataset.REGULAR_GAME_TYPE, nhl_dataset.PLAYOFFS_GAME_TYPE
    """
    data: List[dict] = []
    for season in seasons:
        for game_type in game_types:
            folder_path = nhl.get_folder_path(game_type, season)
            for file_path in glob.glob(os.path.join(folder_path, "*.json")):
                game_data = load_json_dict(file_path)
                if game_data is None:
                    continue
                game_events = get_game_events(game_data, season)
                if len(game_events) > 0:
                    data += game_events

    if len(data) > 0:
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
    else:
        print("No data found!")


def get_period_home_rink_side_right(periods: List[dict]) -> dict:
    """
    Compute a list that indicates for each period if the team that is home is on the right side
    """
    home_rink_side_right = {}
    ref_period = 1
    ref_home = True
    has_valid_rink = False
    for period in periods:
        home_rink_side_right[period["num"]] = None
        if "rinkSide" in period["home"]:
            home_rink_side_right[period["num"]] = period["home"]["rinkSide"] == "right"
            ref_period = period["num"]
            ref_home = period["home"]["rinkSide"] == "right"
            has_valid_rink = True

    if has_valid_rink:
        for period in range(1, len(periods)+7):
            if home_rink_side_right.get(period) == None:
                if period % 2 == ref_period % 2:
                    home_rink_side_right[period] = ref_home
                else:
                    home_rink_side_right[period] = not ref_home
    else:
        home_rink_side_right = None
        print("No valid rink side")
        
    return home_rink_side_right


def get_game_events(game_data: dict, season_year: str) -> List[dict]:
    """
    Get the event data from the raw game data dict

    Arguments:
    - game_data: raw game data received from api
    """
    result = []

    if len(game_data["liveData"]["plays"]["allPlays"]) == 0:
        return result

    home_rink_side_right = get_period_home_rink_side_right(game_data["liveData"]["linescore"]["periods"])

    event = None
    for raw_event in game_data["liveData"]["plays"]["allPlays"]:
        if raw_event["result"]["eventTypeId"] in ["SHOT", "GOAL"]:
            
            if event is not None:
                result.append(event)

            event = {}
            event["game_start_time"] = game_data["gameData"]["datetime"].get("dateTime")
            event["game_end_time"] = game_data["gameData"]["datetime"].get("endDateTime")
            event["season"] = game_data["gameData"]["game"].get("season")
            event["game_id"] = game_data["gameData"]["game"]["pk"]
            event["team_id"] = raw_event["team"]["id"]
            event["team_name"] = raw_event["team"]["name"]
            event["team_tri_code"] = raw_event["team"]["triCode"]
            event["team_link"] = raw_event["team"]["link"]
            event["event_type"] = raw_event["result"]["eventTypeId"]
            event["coordinates_x"] = raw_event["coordinates"].get("x")
            event["coordinates_y"] = raw_event["coordinates"].get("y")

            event_players = get_players_name(raw_event['players'])
            event["shooter_name"] = event_players["shooter_name"]
            event["goalie_name"] = event_players["goalie_name"]

            event["shooter_right_handed"] = get_shooter_right_handed(raw_event['players'], f'./data/{season_year}/players')

            event["shot_type"] = raw_event["result"].get("secondaryType")

            event["goal_empty_net"] = raw_event["result"].get("emptyNet")

            if "strength" in raw_event["result"]:
                event["goal_strength_code"] = raw_event["result"]["strength"]["code"]
            else:
                event["goal_strength_code"] = None
            
            event["period"] = raw_event["about"]["period"]
            event["team_home"] = raw_event["team"]["id"] == game_data["gameData"]["teams"]["home"]["id"]
            if home_rink_side_right == None:
                event["team_rink_side_right"] = None
            else:
                event["team_rink_side_right"] = (event["team_home"] and home_rink_side_right[event["period"]]) or (not event["team_home"] and not home_rink_side_right[event["period"]])

            event["event_id"] = raw_event["about"]["eventId"]

            event["dateTime"] = raw_event["about"]["dateTime"]

            result.append(event)

            event = None
        else:
            event = {}
            event["event_id"] = raw_event["about"]["eventId"]
            event["event_type"] = raw_event["result"]["eventTypeId"]
            event["coordinates_x"] = raw_event["coordinates"].get("x")
            event["coordinates_y"] = raw_event["coordinates"].get("y")
            event["dateTime"] = raw_event["about"]["dateTime"]

    return result


def get_event_features(game_data: dict, raw_event: dict, home_rink_side_right: bool, player_folder_path: str) -> List[dict]:
    """
    Get event feature for a specific `event_data` in the `game_data`
    """

    event = {}

    if raw_event["result"]["eventTypeId"] in ["SHOT", "GOAL"]:
        event["game_start_time"] = game_data["gameData"]["datetime"].get("dateTime")
        event["game_end_time"] = game_data["gameData"]["datetime"].get("endDateTime")
        event["season"] = game_data["gameData"]["game"].get("season")
        event["game_id"] = game_data["gameData"]["game"]["pk"]
        event["team_id"] = raw_event["team"]["id"]
        event["team_name"] = raw_event["team"]["name"]
        event["team_tri_code"] = raw_event["team"]["triCode"]
        event["team_link"] = raw_event["team"]["link"]
        event["event_type"] = raw_event["result"]["eventTypeId"]
        event["coordinates_x"] = raw_event["coordinates"].get("x")
        event["coordinates_y"] = raw_event["coordinates"].get("y")

        event_players = get_players_name(raw_event['players'])
        event["shooter_name"] = event_players["shooter_name"]
        event["goalie_name"] = event_players["goalie_name"]

        event["shooter_right_handed"] = get_shooter_right_handed(raw_event['players'], player_folder_path)

        event["shot_type"] = raw_event["result"].get("secondaryType")

        event["goal_empty_net"] = raw_event["result"].get("emptyNet")

        if "strength" in raw_event["result"]:
            event["goal_strength_code"] = raw_event["result"]["strength"]["code"]
        else:
            event["goal_strength_code"] = None
        
        event["period"] = raw_event["about"]["period"]
        event["team_home"] = raw_event["team"]["id"] == game_data["gameData"]["teams"]["home"]["id"]
        if home_rink_side_right == None:
            event["team_rink_side_right"] = None
        else:
            event["team_rink_side_right"] = (event["team_home"] and home_rink_side_right[event["period"]]) or (not event["team_home"] and not home_rink_side_right[event["period"]])

        event["event_id"] = raw_event["about"]["eventId"]

        event["dateTime"] = raw_event["about"]["dateTime"]
    else:
        event["event_id"] = raw_event["about"]["eventId"]
        event["event_type"] = raw_event["result"]["eventTypeId"]
        event["coordinates_x"] = raw_event["coordinates"].get("x")
        event["coordinates_y"] = raw_event["coordinates"].get("y")
        event["dateTime"] = raw_event["about"]["dateTime"]

    return event


def get_players_name(players: List[dict]) -> dict:
    """
    Get the name and role of the players from the player list

    Arguments:
    - players: the player list from a game event
    """
    result = {"shooter_name": "", "goalie_name": ""}
    for player in players:
        if player["playerType"] in ["Scorer", "Shooter"]:
            result["shooter_name"] = player["player"]["fullName"]
        elif player["playerType"] == "Goalie":
            result["goalie_name"] = player["player"]["fullName"]
    return result


def get_shooter_right_handed(players: List[dict], folder_path: str) -> bool:
    """
    Check if the shooter is right handed
    
    - players: the list of players on the event
    - folder_path: the folder to save the player json downloaded from the api
    """
    right_handed = False
    for player in players:
        if player["playerType"] in ["Scorer", "Shooter"]:
            download_player_data(folder_path, player["player"]['id'], player["player"]['link'])
            player_full_data = load_json_dict(join(folder_path, f"{player['player']['id']}.json"))
            if player_full_data is not None:
                if len(player_full_data['people']) == 0:
                    print('no player info found')
                else:
                    right_handed = player_full_data['people'][0]['shootsCatches'] == 'R'
                if len(player_full_data['people']) > 1:
                    print('Multiple player info found')

    return right_handed


def download_player_data(folder_path: str, player_id: str, player_link: str):
    """
    Download player data if not already exisiting in the folder

    Arguments:
    - folder_path: where to save the data
    - player_id: the id of the player 
    - player_link: the relative link where to fetch the player info
    """
    file_path = join(folder_path, f"{player_id}.json")

    if exists(file_path):
        return

    Path(folder_path).mkdir(parents=True, exist_ok=True)

    try:
        url = f"{nhl_dataset.NHL_API_DOMAIN}{player_link}"
        r = requests.get(url)
        if r.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(r.content)
        else:
            print(f"Could not download player with id: {player_id} - status code: {r.status_code}")
    except Exception as e :
        print("====================================================================")
        print(f"Could not download player with id: {player_id}")
        print(e)
        print("====================================================================")


def load_json_dict(file_path: str) -> dict:
    """
    Load the json data from the file_path into a dict

    Arguments:
    - file_path: the json file
    """
    result = None
    try:
        with open(file_path) as f:
            result = json.load(f)
    except:
        print(f"Could not load json at file {file_path}")
    return result


if __name__ == "__main__":
    dataset = nhl_dataset.NhlDataset()
    seasons = [2015, 2016, 2017, 2018, 2019]

    for season in seasons:
        dataset.load_regular_season(season)
    
    convert_raw_data_to_panda_csv(dataset, './data/train.csv', [2015, 2016, 2017, 2018], [nhl_dataset.REGULAR_GAME_TYPE])

    convert_raw_data_to_panda_csv(dataset, './data/test.csv', [2019], [nhl_dataset.REGULAR_GAME_TYPE])