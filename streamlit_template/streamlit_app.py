import streamlit as st
import pandas as pd
import numpy as np
import requests
import json



st.title("Hockey visualization App")
with st.form(key ='Form1'):
   
    with st.sidebar:
        # TODO: Add input for the sidebar

        workspace = st.sidebar.markdown("Workspace")
        Model = st.sidebar.markdown("Model")
        Version = st.sidebar.markdown("Version")
        
        getmodel = st.form_submit_button(label = 'Get Models')
        pass

with st.container():
    # TODO: Add Game ID input
    

    form = st.form(key='GameID')
    gameid = form.number_input('Enter GameID',step=None,value=0)
    submit_button = form.form_submit_button(label='Ping game')
    if submit_button:
        data = requests.get('https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/'.format(gameid)).json()
        td_home = data["liveData"]["boxscore"]["teams"]["home"]["teamStats"]["teamSkaterStats"]
        home = data["gameData"]["teams"]["home"]["name"]
        away = data["gameData"]["teams"]["away"]["name"]
        st.write('Game ', gameid , home , ' vs ',away)
        
    pass

with st.container():
    # TODO: Add Game info and predictions
    
    pass

with st.container():
    # TODO: Add data used for predictions
    pass