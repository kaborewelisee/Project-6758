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


def create_heatmap(data):
    '''
    Filters and converts the coordinates data to be used for a given team and season.

    Args:
        data: df of the data for a given team and season

    Returns:
        x: coordinates in x to be used
        y: coordinates in y to be used

    '''
    # print(data[['coordinates_x', 'coordinates_y', 'team_home', 'team_rink_side_right']])
    data['coordinates_x'] = np.where(data['team_rink_side_right'], -data['coordinates_x'], data['coordinates_x'])
    data['coordinates_y'] = np.where(data['team_rink_side_right'], -data['coordinates_y'], data['coordinates_y'])

    data = data[data['coordinates_x'] >= 0]

    x = data['coordinates_x'].values
    y = data['coordinates_y'].values
    # h, _, _, _ = plt.hist2d(y, x, bins=[np.arange(-40, 40, 1), np.arange(0, 90, 1)])

    return x, y


def shoots_visual():
    '''
    Makes the advanced visualisation plot and saves it to plot/shootings.html
    '''

    df = pd.read_csv('data/df.csv')
    df = df[~df['team_rink_side_right'].isnull()]

    df = df[df['team_tri_code'].isin(['MTL', 'TOR'])]
    df = df[df['season'].isin([20192020, 20202021])]

    # teams = df['team_name'].unique()
    # seasons = df['season'].unique()
    traces = []
    menu_items = []
    for (team_nm, season), data in df.groupby(['team_name', 'season']):
        menu_items.append(f'{team_nm}-{season}')
        # heatmap[team_nm][season_id] = create_heatmap(data)
        x, y = create_heatmap(data)
        fig = ff.create_2d_density(y, x)
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
    layout = dict(updatemenus=updatemenus, title=menu_items[0])
    fig = dict(data=traces, layout=layout)

    pio.write_html(fig, 'plots/shootings.html')
    # iplot(fig)


if __name__ == "__main__":
    shoots_visual()
