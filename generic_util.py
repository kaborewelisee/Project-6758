import numpy as np
import pandas as pd


NET_ABSOLUTE_COORD_X = 89
NET_COORD_Y = 0


def get_shot_distance(x: int, y: int, team_rink_side_right: bool) -> int:
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


def get_shot_angle(x: int, y: int, team_rink_side_right: bool, shooter_right_handed: bool) -> int:
    """
    Calculate the shot angle

    Arguments:
    - x: shooter x coordinate on ice
    - y: shooter y coordinate on ice
    - team_rink_side_right: indicates if the shhoter rink is right
    - shooter_right_handed: indicates if shooter is right handed
    """

    angle = 0

    if y == 0:
        angle = np.pi / 2
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

    return angle



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
