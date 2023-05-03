"""
This is a library that can be used by other files to update the session_state.

Author: Swathi Natarajan <swathinatara@umass.edu>
Created: March 27, 2023
Version: 1.0
"""
import yaml
import os
import streamlit as st

#
# This module updates session_state["config_session"] with the dictionary<key, value> pairs from scale_profile.yaml.
# Input: None 
# Output: none
# USAGE: 
#    Before calling this function, the caller is responsible for checking if the session_state["config_session"] is present.

def fUpdateSessionDefaultProfile():
    with open('./scale_profile.yaml', 'r') as f:
        scaleProfile_session = yaml.safe_load(f)
        st.session_state["config_session"] = scaleProfile_session
    return