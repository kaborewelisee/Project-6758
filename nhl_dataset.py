from os.path import exists
from os.path import join
from pathlib import Path
import requests
from collections.abc import Generator

REGULAR_GAME_TYPE = "02"
PLAYOFFS_GAME_TYPE = "03"

NHL_API_DOMAIN = "https://statsapi.web.nhl.com"
LIVE_DATA_ENDPOINT = "/api/v1/game/{game_id}/feed/live/"


class NhlDataset:
    """
    Utility class to downlaod and acces NHL game play data
    """

    def __init__(self, cache_path = './data') -> None:
        """
        
        Arguments:
        - cache_path: folder path where data will be downloaded. When not specified, a data folder is created in current folder
        """
        self.cache_path = cache_path


    def load_all(self, *seasons: int) -> None:
        """
        Load regular season and playoffs games play data for multiple seasons

        Arguments:
        - season: the start year of each season to load. For example, for 2016-17 it is 2016.
        """
        for season in seasons:
            self.load_regular_season(season)
            self.load_playoffs(season)
    

    def get_folder_name(self, game_type: str) -> str:
        """
        Get folder name for a specific game type

        Arguments:
        - game_type: the game type, see constants: REGULAR_GAME_TYPE, PLAYOFFS_GAME_TYPE
        """
        if game_type == REGULAR_GAME_TYPE:
            return "regular"
        elif game_type == PLAYOFFS_GAME_TYPE:
            return "playoffs"
        else:
            return "unknown"


    def get_folder_path(self, game_type: str, season: int) -> str:
        """
        Get the path of the folder to save the season data for a specific game type

        Arguments:
        - game_type: the game type, see constants: REGULAR_GAME_TYPE, PLAYOFFS_GAME_TYPE
        - season: the start year of each season to load. For example, for 2016-17 it is 2016.
        """
        folder_name = self.get_folder_name(game_type)
        return join(self.cache_path, str(season), folder_name)


    def format_game_id(self, season: int, game_type: str, game_number: str) -> str:
        """
        Format the game id

        Arguments:
        - season: the start year of each season to load. For example, for 2016-17 it is 2016
        - game_type: the game type, see constants: REGULAR_GAME_TYPE, PLAYOFFS_GAME_TYPE
        - game_number: the number of the game
        """
        return f"{season}{game_type}{game_number}"

    
    def get_regular_season_games_count(self, season: int):
        """
        Get the maximum number of possible games for a season

        Arguments:
        - season: the start year of each season to load. For example, for 2016-17 it is 2016
        """
        if season >= 2017:
            return 1271
        else:
            return 1230


    def download(self, game_id: str, folder_path: str) -> None:
        """
        Download data for a specific game if not already exisiting in the folder

        Arguments:
        - game_id: the id of the game 
        - folder_path: where to save the data
        """
        file_path = join(folder_path, f"{game_id}.json")

        if exists(file_path):
            return

        try:
            url = f"{NHL_API_DOMAIN}{LIVE_DATA_ENDPOINT.format(game_id = game_id)}"
            r = requests.get(url)
            if r.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(r.content)
            else:
                print(f"Could not download game with id: {game_id} - status code: {r.status_code}")
        except Exception as e :
            print("====================================================================")
            print(f"Could not download game with id: {game_id}")
            print(e)
            print("====================================================================")


    def load_season_games(self, season: int, game_type: str, game_id_generator: lambda int: Generator[str, str, str]) -> None:
        """
        Load games play data for a specific season, game_type and game_id provided by the generator

        Arguments:
        - season: the start year of the season. For example, for 2016-17 it is 2016
        - game_type: the game type, see constants: REGULAR_GAME_TYPE, PLAYOFFS_GAME_TYPE
        - game_id_generator: a function that generate the id of the game to download
        """
        folder_path = self.get_folder_path(game_type, season)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        for game_id in game_id_generator(season):
            self.download(game_id, folder_path)


    def regular_season_game_id_generator(self, season: int) -> Generator[str, str, str]:
        """
        Generate all possible game id for a regular season

        Arguments:
        - season: the start year of the season. For example, for 2016-17 it is 2016
        """
        for i in range(self.get_regular_season_games_count(season)):
            game_id = self.format_game_id(season, REGULAR_GAME_TYPE, f"{i+1:04d}")
            yield game_id


    def playoffs_game_id_generator(self, season: int) -> Generator[str, str, str]:
        """
        Generate all possible game id for a playoffs season

        Arguments:
        - season: the start year of the season. For example, for 2016-17 it is 2016
        """
        pass


    def load_regular_season(self, season: int) -> None:
        """
        Load regular season games play data for a specific season

        Arguments:
        - season: the start year of the season. For example, for 2016-17 it is 2016
        """
        self.load_season_games(season, REGULAR_GAME_TYPE, self.regular_season_game_id_generator)


    def load_playoffs(self, season: int) -> None:
        """
        Load playoffs games play data for a specific season

        Arguments:
        - season: the start year of the season. For example, for 2016-17 it is 2016
        """
        # self.load_season_games(season, PLAYOFFS_GAME_TYPE, self.playoffs_game_id_generator)
        pass



if __name__ == "__main__":
    dataset = NhlDataset()
    dataset.load_all(2016, 2017, 2018, 2019, 2020)