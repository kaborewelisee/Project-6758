from typing import List
import nhl_dataset
import glob
import os
import pandas as pd
import json

def convert_raw_data_to_panda_csv(nhl: nhl_dataset.NhlDataset, csv_path: str, seasons: List[int], game_types: List[str] = [nhl_dataset.REGULAR_GAME_TYPE, nhl_dataset.PLAYOFFS_GAME_TYPE]):
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


def get_game_events(game_data: dict) -> List[dict]:
    result = []
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

            result.append(event)
    return result


def get_players_name(players: List[dict]) -> dict:
    result = {"shooter_name": "", "goalie_name": ""}
    for player in players:
        if player["playerType"] in ["Scorer", "Shooter"]:
            result["shooter_name"] = player["player"]["fullName"]
        elif player["playerType"] == "Goalie":
            result["goalie_name"] = player["player"]["fullName"]
    return result


def load_json_dict(file_path: str) -> dict:
    result = None
    try:
        with open(file_path) as f:
            result = json.load(f)
    except:
        print(f"Could not load json at file {file_path}")
    return result


if __name__ == "__main__":
    dataset = nhl_dataset.NhlDataset()
    dataset.load_all(2016, 2017, 2018, 2019)
    seasons = [2016, 2017, 2018, 2019, 2020]
    convert_raw_data_to_panda_csv(dataset, './data/df-1.csv', seasons)