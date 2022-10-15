from typing import List
import nhl_dataset
import glob
import os
import pandas as pd
import json

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
                game_events = get_game_events(game_data)
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


def get_game_events(game_data: dict) -> List[dict]:
    """
    Get the event data from the raw game data dict

    Arguments:
    - game_data: raw game data received from api
    """
    result = []

    if len(game_data["liveData"]["plays"]["allPlays"]) == 0:
        return result

    home_rink_side_right = get_period_home_rink_side_right(game_data["liveData"]["linescore"]["periods"])

    for raw_event in game_data["liveData"]["plays"]["allPlays"]:
        if raw_event["result"]["eventTypeId"] in ["SHOT", "GOAL"]:
            event = {}
            event["game_start_time"] = game_data["gameData"]["datetime"].get("dateTime")
            event["game_end_time"] = game_data["gameData"]["datetime"].get("endDateTime")
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


            result.append(event)
    return result


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
    #dataset.load_all(2016, 2017, 2018, 2019, 2020)
    seasons = [2016, 2017, 2018, 2019, 2020]
    convert_raw_data_to_panda_csv(dataset, './data/df.csv', seasons)