import streamlit as st
import pandas as pd
import numpy as np
import requests
import json


st.title("Hockey visualization App")
with st.form(key='Form1'):
   
    with st.sidebar:
        # TODO: Add input for the sidebar

        workspace = st.sidebar.markdown("Workspace")
        Model = st.sidebar.markdown("Model")
        Version = st.sidebar.markdown("Version")
        
        getmodel = st.form_submit_button(label='Get Models')
        pass

with st.container():
    # TODO: Add Game ID input

    form = st.form(key='GameID')
    gameid = form.number_input('Enter GameID', step=None, value=0)
    submit_button = form.form_submit_button(label='Ping game')
    # gameid = st.text_input('GameID')

    if submit_button:
    # if st.button('Ping game'):

        # TODO get data from NHL from the game_id
        #  - Home team
        #  - Away team
        #  - Period #
        #  - Temps restant periode
        #  - Actual score

        data = requests.get('https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/'.format(gameid)).json()
        print(data)
        # td_home = data["liveData"]["boxscore"]["teams"]["home"]["teamStats"]["teamSkaterStats"]
        # home = data["gameData"]["teams"]["home"]["name"]
        # away = data["gameData"]["teams"]["away"]["name"]
        home = 'Canadiens'
        away = 'Toronto'
        st.write('Game ', gameid, home, ' vs ', away)
        actual_goals = {
            'home': 2,
            'away': 4,
        }

        # TODO get data from client API
        #  - Predictions
        #  - Features
        events = pd.DataFrame(
            [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            columns=['feature0', 'feature1', 'feature2', 'feature3', 'feature4', 'prediction']
        )
        pred_goals = {
            'home': sum([0.33, 0.45, 0.11, 0.89]),
            'away': sum([0.63, 0.95, 0.31, 0.79]),
        }
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label=f"{home} - Expected Goals (actual)",
                value=f"{pred_goals['home']} ({actual_goals['home']})",
                delta=pred_goals['home']-actual_goals['home']
            )
        with col2:
            st.metric(
                label=f"{away} - Expected Goals (actual)",
                value=f"{pred_goals['away']} ({actual_goals['away']})",
                delta=pred_goals['away']-actual_goals['away']
            )

        st.table(data=events)

with st.container():
    # TODO: Add Game info and predictions
    pass

with st.container():
    # TODO: Add data used for predictions
    pass
