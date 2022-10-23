import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
from plotly.offline import iplot
import kaleido
import os
import matplotlib.pyplot as plt
import numpy as np

if not os.path.exists("plots"):
    os.mkdir("plots")


def create_heatmap(data, season_avg_hourly):
    """
    Filters and converts the coordinates data to be used for a given team and season.

    Args:
        data: df of the data for a given team and season
        season_avg: hist of the season average for a given season

    Returns:
        diff: difference between the hourly shooting rate of a given team vs the league average
        x_labels: x labels
        y_labels: y labels

    """
    # print(data[['coordinates_x', 'coordinates_y', 'team_home', 'team_rink_side_right']])
    data['coordinates_x'] = np.where(data['team_rink_side_right'], -data['coordinates_x'], data['coordinates_x'])
    data['coordinates_y'] = np.where(data['team_rink_side_right'], -data['coordinates_y'], data['coordinates_y'])

    data = data[data['coordinates_x'] >= 0]

    x = data['coordinates_x'].values
    y = data['coordinates_y'].values
    team_avg, x_labels, y_labels, _ = plt.hist2d(y, x, bins=[np.arange(-40, 40, 2), np.arange(0, 90, 2)])

    # Make the realtive difference between the team average and the league
    nb_games = len(data['game_id'].unique())
    team_avg_hourly = team_avg / nb_games
    diff = team_avg_hourly - season_avg_hourly

    return diff, x_labels, y_labels


def make_season_avg(data):
    """
    Makes the season average of shoots per hour for all teams for a given season

    Args:
        data: season data for all teams

    Returns:
        season_avg: 2D histogram of the season average of shoots per hour in the offencive zone for all teams
    """

    data['coordinates_x'] = np.where(data['team_rink_side_right'], -data['coordinates_x'], data['coordinates_x'])
    data['coordinates_y'] = np.where(data['team_rink_side_right'], -data['coordinates_y'], data['coordinates_y'])

    data = data[data['coordinates_x'] >= 0]

    x = data['coordinates_x'].values
    y = data['coordinates_y'].values
    season_avg, _, _, _ = plt.hist2d(y, x, bins=[np.arange(-40, 40, 2), np.arange(0, 90, 2)])

    nb_games = len(data['game_id'].unique())
    season_avg_hourly = season_avg / nb_games

    return season_avg_hourly


def shoots_visual():
    """
    Makes the advanced visualisation plot and saves it to plot/shootings.html
    """

    df = pd.read_csv('data/df.csv')
    df = df[~df['team_rink_side_right'].isnull()]

    # df = df[df['team_tri_code'].isin(['MTL', 'TOR'])]
    # df = df[df['season'].isin([20192020, 20202021])]

    # teams = df['team_name'].unique()
    seasons = df['season'].unique()
    seasons_avg = {}
    for season in seasons:
        seasons_avg[season] = make_season_avg(df[df['season'] == season])

    traces = []
    menu_items = []
    for (team_nm, season), data in df.groupby(['team_name', 'season']):
        menu_items.append(f'{team_nm}-{season}')
        # heatmap[team_nm][season_id] = create_heatmap(data)
        diff_heatmap, x_labels, y_labels = create_heatmap(data, seasons_avg[season])
        # fig = ff.create_2d_density(y, x)
        fig = ff.create_annotated_heatmap(np.transpose(diff_heatmap), x=list(x_labels)[:-1], y=list(y_labels)[:-1], showscale=True)
        traces.append(fig.to_dict()["data"][0])

    # Create figure
    # fig = go.Figure()

    # # Add surface trace
    # fig.add_trace(go.Heatmap(z=df.values.tolist(), colorscale="Viridis"))

    # # Update plot sizing
    # fig.update_layout(
    #     width=800,
    #     height=900,
    #     autosize=False,
    #     margin=dict(t=100, b=0, l=0, r=0),
    # )

    # # Update 3D scene options
    # fig.update_scenes(
    #     aspectratio=dict(x=1, y=1, z=0.7),
    #     aspectmode="manual"
    # )

    # Add dropdowns
    # button_layer_1_height = 1.08
    # fig.update_layout(
    #     updatemenus=[
    #         dict(
    #             buttons=list([
    #                 dict(
    #                     args=team_nm,
    #                     label=team_nm,
    #                     method="restyle"
    #                 )
    #                 for team_nm in teams
    #             ]),
    #             direction="down",
    #             pad={"r": 10, "t": 10},
    #             showactive=True,
    #             x=0.1,
    #             xanchor="left",
    #             y=button_layer_1_height,
    #             yanchor="top"
    #         ),
    #         dict(
    #             buttons=list([
    #                 dict(
    #                     args=season,
    #                     label=season,
    #                     method="restyle"
    #                 )
    #                 for season in seasons
    #             ]),
    #             direction="down",
    #             pad={"r": 10, "t": 10},
    #             showactive=True,
    #             x=0.37,
    #             xanchor="left",
    #             y=button_layer_1_height,
    #             yanchor="top"
    #         )
    #     ]
    # )

    # fig.update_layout(
    #     annotations=[
    #         dict(text="colorscale", x=0, xref="paper", y=1.06, yref="paper",
    #                              align="left", showarrow=False),
    #         dict(text="Reverse<br>Colorscale", x=0.25, xref="paper", y=1.07,
    #                              yref="paper", showarrow=False),
    #         dict(text="Lines", x=0.54, xref="paper", y=1.06, yref="paper",
    #                              showarrow=False)
    #     ])

    buttons = []
    for i, menu_item in enumerate(menu_items):
        visibility = [i == j for j in range(len(menu_items))]
        button = dict(
            label=menu_item,
            method='update',
            args=[{'visible': visibility},
                  {'title': menu_item}])
        buttons.append(button)

    updatemenus = list([
        dict(buttons=buttons)
    ])
    team_str = menu_items[0].split('-')[0]
    season_str = f"Season: {menu_items[0].split('-')[1]}"
    text_desc = f'{team_str} Offence<br>' \
                f'Season {season_str}<br>' \
                f'Unblocked Shot Rates, relative to League Average of the Season' \
                # f'{}<br>' \
                # f'{}'
    layout = dict(updatemenus=updatemenus, title=text_desc, showlegend=True)
    fig = dict(data=traces, layout=layout)

    pio.write_html(fig, 'plots/shootings.html')
    # iplot(fig)


if __name__ == "__main__":
    shoots_visual()
