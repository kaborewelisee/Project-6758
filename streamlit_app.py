import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import datetime
import sys
from ift6758.client import game_client, serving_client

########################################
# Partie client
# For local testing
#pinger = game_client.GameClient("./predictions", "127.0.0.1", 5000)
#serving = serving_client.ServingClient(ip="127.0.0.1", port=5000)

# For Docker 
pinger = game_client.GameClient("./predictions", "flask", 5000)
serving = serving_client.ServingClient(ip="flask",port=5000)



global memoryGamet1
global memoryGamet2
global memoryGameID
global homeOrAway
memoryGamet1=0
memoryGamet2=0
memoryGameID=0
homeOrAway=[]



st.title("Hockey visualization App")

with st.form(key='Form2'):
   
    with st.sidebar:
        # TODO: Add input for the sidebar
        workspace1 = st.selectbox('Workspace', ('ift6758-22-milestone-2','test'))
        Model1 = st.selectbox('Model', ('question-6-random-forest-classifier-base','xgboost-task5-model'))
        Version1 = st.selectbox('Version', ('3.0.0',' '))
        
        getmodel1 = st.form_submit_button(label='Get Models')
        pass

with st.container():
    form1 = st.form(key='GameID')
    gameid1 = form1.number_input('Enter GameID', step=None, value=2021020312)
    ping_game = form1.form_submit_button(label='Ping games')
        
 
pass

if getmodel1:
            #serving.download_registry_model('ift6758-22-milestone-2', 'question-6-random-forest-classifier-base', '1.0.0')
            
            dic= serving.download_registry_model(workspace1,Model1,Version1)
            
            st.write(dic['message'])
            #st.write("Model ",Version1, " Version ",Model1," Workspace ",workspace1," has been uploaded")
            

            





if ping_game:

        

        tag = workspace1+Model1+Version1
        data = requests.get('https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/'.format(gameid1)).json()
        game=(pinger.ping(gameid1,tag))
        period1 = game["period"].iloc[-1]
        s = game.iloc[-1]["game_elapsed_time"]
        
        timet1 = str(datetime.timedelta(seconds=int(s)))
        
        #timet1 = str(game.iloc[-1]["game_period_seconds"]).split(":")

        
       # print(data["liveData"]["plays"]["allPlays"])
       # period_no = data["liveData"]["plays"]["allPlays"][-1]["about"]["period"]
       # periodTimeRemaining = data["liveData"]["plays"]["allPlays"][-1]["about"]["periodTimeRemaining"]
       # # td_home = data["liveData"]["boxscore"]["teams"]["home"]["teamStats"]["teamSkaterStats"]
       # home = data["gameData"]["teams"]["home"]["name"]
       # away = data["gameData"]["teams"]["away"]["name"]
       # st.subheader(f'Game {gameid1}: {away} @ {home}')
       # st.write(f'Period {period_no} - {periodTimeRemaining} left')
       # actual_goals = {
       #    'home': data["liveData"]["plays"]["allPlays"][-1]["about"]["goals"]["home"],
       #    'away': data["liveData"]["plays"]["allPlays"][-1]["about"]["goals"]["away"],
       #                }
        
        timeLeftt1 = data["liveData"]["plays"]["allPlays"][-1]["about"]["periodTimeRemaining"]

        # get Team Names
        team1 = data["gameData"]["teams"]["home"]["name"]
        team2 = data["gameData"]["teams"]["away"]["name"]

        # Separate dataframes for each team
        gamet1= game[game['team_name']==team1]
        gamet2= game[game['team_name']==team2]
        
        #Set up memory game and ID
        if type(memoryGameID)==type(0):
            #if no memory yet, set memoryGameID
            memoryGameID=str(gameid1)
        if type(memoryGamet1)==type(0):
            #if no memory yet, set memoryGame
            memoryGamet1=gamet1
            memoryGamet2=gamet2
        else:
            if memoryGameID!=str(gameid1):
                #if the memoryGameID does not correspond to the input gameID, it means that
                #we are dealing with a new game. 
                memoryGamet1=gamet1
                memoryGamet2=gamet2
                memoryGameID=str(gameid1)
            else:
                #This is if the new events belong to the same game
                memoryGamet1.append(gamet1,ignore_index=True)
                memoryGamet2.append(gamet2,ignore_index=True)


        
        
        st.subheader(f'Game {gameid1}: {team1} @ {team2}')
        st.write(f'Period {period1} - {timeLeftt1} left')
        actual_goals = {
            'home': data["liveData"]["plays"]["allPlays"][-1]["about"]["goals"]["home"],
            'away': data["liveData"]["plays"]["allPlays"][-1]["about"]["goals"]["away"],
        }
        
        #  Corresponding XG for each teams
        XGt1 = gamet1[['goal_probability']]
       
       
        #XGt1 = XGt1.drop_duplicates(subset='goal_probability', keep='first')

        XGt2 = gamet2[['goal_probability']]
        #XGt2 = XGt2.drop_duplicates(subset='goal_probability', keep='first')
        # Sum
        sumt1 = float(XGt1.sum())
        sumt2 = float(XGt2.sum())
        
        # Dataframe with only goals for each team
        Gt1 = gamet1[gamet1['event_type']=='GOAL']
        
        #Gt1 = Gt1.drop_duplicates(subset='event_id', keep='first')

        Gt2 = gamet2[gamet2['event_type']=='GOAL']
        
        #Gt2 = Gt2.drop_duplicates(subset='event_id', keep='first')
        # Sum of goals for each team
        goalst1 = int(len(Gt1.index))
        goalst2 = int(len(Gt2.index))

        delta1 = sumt1 - goalst1
        delta2 = sumt2 - goalst2
        

      # st.subheader('Features data and predictions')
      # col1, col2 = st.columns(2)
      # with col1:
      #     st.metric(
      #         label=f"{home} - Expected Goals (actual)",
      #         value=f"{pred_goals['home']} ({actual_goals['home']})",
      #         delta=pred_goals['home']-actual_goals['home']
      #     )
      # with col2:
      #     st.metric(
      #     label=f"{away} - Expected Goals (actual)",
      #         value=f"{pred_goals['away']} ({actual_goals['away']})",
      #         delta=pred_goals['away']-actual_goals['away']
      #     )

    # Probleme: les stats ne se reinstalise pas apres chaque ping. 
        st.subheader('Features data and predictions')
        col1, col2 = st.columns(2)
        col1.metric(
               label=f"{team1} - Expected Goals (actual)",
               value=f"{'%.2f' % sumt1} ({goalst1})",
               delta='%.2f' % delta1
           )
        with col2:
             st.metric(
                 label=f"{team2} - Expected Goals (actual)",
                 value=f"{'%.2f' % sumt2} ({goalst2})",
                 delta='%.2f' % delta2
             )
        #events = pd.DataFrame(
        #    [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        #   columns=['feature0', 'feature1', 'feature2', 'feature3', 'feature4', 'prediction']
        #)
        #pred_goals = {
        #    'home': sum([0.33, 0.45, 0.11, 0.89]),
        #    'away': sum([0.63, 0.95, 0.31, 0.79]),
        #}
        

        #st.table(data=events)
        
        #game = game.drop_duplicates(subset='event_id', keep='first')
        df1 =  game[['coordinates_x', 'coordinates_y', 'period', 'game_elapsed_time', 'shot_distance', 'shot_angle', 'hand_based_shot_angle', 'empty_net', 'last_coordinates_x', 'last_coordinates_y', 'time_since_last_event', 'distance_from_last_event', 'rebond', 'speed_from_last_event', 'shot_angle_change', 'ShotType_Backhand', 'ShotType_Deflected', 'ShotType_Slap Shot', 'ShotType_Snap Shot', 'ShotType_Tip-In', 'ShotType_Wrap-around', 'ShotType_Wrist Shot','goal_probability']]
        st.dataframe(df1)




#with st.container():
#    form = st.form(key='GameID')
#    gameid = form.number_input('Enter GameID', step=None, value=0)
#    submit_button = form.form_submit_button(label='Ping game')
#    # gameid = st.text_input('GameID')
#
#    if submit_button:
#     if st.button('Ping game'):
#
#        data = requests.get('https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/'.format(gameid)).json()
#        print(data["liveData"]["plays"]["allPlays"])
#        period_no = data["liveData"]["plays"]["allPlays"][-1]["about"]["period"]
#        periodTimeRemaining = data["liveData"]["plays"]["allPlays"][-1]["about"]["periodTimeRemaining"]
#        # td_home = data["liveData"]["boxscore"]["teams"]["home"]["teamStats"]["teamSkaterStats"]
#        home = data["gameData"]["teams"]["home"]["name"]
#        away = data["gameData"]["teams"]["away"]["name"]
#        st.subheader(f'Game {gameid}: {away} @ {home}')
#        st.write(f'Period {period_no} - {periodTimeRemaining} left')
#        actual_goals = {
#            'home': data["liveData"]["plays"]["allPlays"][-1]["about"]["goals"]["home"],
#            'away': data["liveData"]["plays"]["allPlays"][-1]["about"]["goals"]["away"],
#        }
#
#        # TODO get data from client API
#        #  - Predictions
#        #  - Features
#
