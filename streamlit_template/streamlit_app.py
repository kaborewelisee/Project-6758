import streamlit as st
import pandas as pd
import numpy as np


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
    form.text_input(label='Enter GameID')
    submit_button = form.form_submit_button(label='Ping')
    pass

with st.container():
    # TODO: Add Game info and predictions
    pass

with st.container():
    # TODO: Add data used for predictions
    pass