import json
import requests
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """

        input = {}

        for feature in self.features:
            input[feature] = X[feature].values.tolist()

        r = requests.post(
            f"{self.base_url}/predict", 
            json=input
        )

        response = { 'success': False, 'data': None, 'message': 'Unexpected error!'  }

        if (r.status_code == 200):
            response['success'] = True
            response['data'] = pd.DataFrame.from_dict(r.json())
        elif (r.status_code == 404 or r.status_code == 400):
            response['success'] = False
            response['message'] = r.json()['message']
            
        return response


    def logs(self) -> dict:
        """Get server logs"""

        r = requests.get(f"{self.base_url}/logs")

        response = { 'success': False, 'data': None, 'message': '' }

        if r.status_code == 200:
            response['success'] = True
            response['data'] = r.json()
        else:
            response['success'] = False
            response['message'] = 'Unexpected error!'

        return response
        

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        r = requests.post(
            f"{self.base_url}/download_registry_model", 
            json={
                'workspace' : workspace,
                'model' : model,
                'version' : version
            }
        )

        response = { 'success': False, 'message': '' }

        if r.status_code == 200:
            response['success'] = True
            response['message'] = r.json()
        elif r.status_code == 404:
            response['success'] = False
            response['message'] = r.json()['message']
        else:
            response['success'] = False
            response['message'] = 'Unexpected error!'

        return response


if __name__ == "__main__":
    client = ServingClient("127.0.0.1", 5000, ['coordinates_x', 'coordinates_y', 'period', 'game_period_seconds', 'game_elapsed_time', 'shot_distance', 'shot_angle', 'hand_based_shot_angle', 'empty_net', 'last_coordinates_x', 'last_coordinates_y', 'time_since_last_event', 'distance_from_last_event', 'rebond', 'speed_from_last_event', 'shot_angle_change', 'ShotType_Backhand', 'ShotType_Deflected', 'ShotType_Slap Shot', 'ShotType_Snap Shot', 'ShotType_Tip-In', 'ShotType_Wrap-around', 'ShotType_Wrist Shot'])
    download = client.download_registry_model('ift6758-22-milestone-2', 'question-6-random-forest-classifier-base', '1.0.0')
    print(download)
    input = {
        "coordinates_x": [-69.0, 68.0],
        "coordinates_y": [-4.0, -27.0],
        "period": [2, 3],
        "game_period_seconds": [9283.0, 9987.0],
        "game_elapsed_time": [4955.0, 7637.0],
        "shot_distance": [20.396078054371134, 34.20526275297414],
        "shot_angle": [-11.309932474020211, 52.1250163489018],
        "hand_based_shot_angle": [101.3099324740202, 37.874983651098205],
        "empty_net": [0, 0],
        "last_coordinates_x": [-9.0, 83.0],
        "last_coordinates_y": [-3.0, -6.0],
        "time_since_last_event": [38.0, 17.0],
        "distance_from_last_event": [60.00833275470999, 25.80697580112788],
        "rebond": [0, 1],
        "speed_from_last_event": [1.579166651439737, 1.5180574000663458],
        "shot_angle_change": [0.0, 7.125016348901795],
        "ShotType_Backhand": [0, 0],
        "ShotType_Deflected": [0, 0],
        "ShotType_Slap Shot": [0, 0],
        "ShotType_Snap Shot": [0, 0],
        "ShotType_Tip-In": [0, 0],
        "ShotType_Wrap-around": [0, 0],
        "ShotType_Wrist Shot": [1, 1]
    }
    predict = client.predict(pd.DataFrame.from_dict(input))
    print(predict)
    logs = client.logs()
    print(logs)

