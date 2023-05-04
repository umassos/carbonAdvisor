"""
This represents a page in carbonAdvisor to upload profile or trace files to the system.
It takes in a profile yaml file as input, validates it and stores the information in a session.
The session storage is then later accessed by respective files to get "task_profile".

Author: Swathi Natarajan <swathinatara@umass.edu>
Created: March 15, 2023
Version: 1.0
"""

import streamlit as st
import yaml
from Update_Session import fUpdateSessionDefaultProfile


st.markdown("## Upload profile files")

#--------------------------------Implemented the profile upload using session state---------------------------------------#

config_session = st.file_uploader("Please upload a file in the right format")

# Call the library to update the session state "Config_session" if it is not available.
if "config_session" not in st.session_state:
    fUpdateSessionDefaultProfile()

if config_session is not None:
    temp = yaml.safe_load(config_session)
    
    try:
        #validation of the new uploaded yaml file:
        for model in temp:
            #1. Check if the task profile in the uploaded file exists in schema
            assert model not in st.session_state["config_session"].keys(), "Keys already found in the existing profile"
            
            #2. For a particular task profile, check if 'instance' exists and has the value same as the type of schema
            assert 'instance' in temp[model], "instance type key is missing in the uploaded file" 
            ## todo: check if instance is correct
            
            #3.For a particular task profile, check if 'replicas' exists and has the same number of replicas as schema.
            assert 'replicas' in temp[model], "replicas key is missing in the uploaded file"
            assert temp[model]['replicas'] is not None, "Replica list is empty. Minimum 1 replicas should be present"
            assert list(range(min(temp[model]['replicas'].keys()), max(temp[model]['replicas'].keys())+1)) == list(temp[model]['replicas'].keys()), "The replica numbers are not continuous"

            #4. For each replica, check if 'power' and 'throughput' exists and has the value same as the type of schema["densenet121" replica 1 is taken as schema reference] TBD:: This can be improved by introducing a schema file
            for replica in temp[model]['replicas']:
                assert 'power' in temp[model]['replicas'][replica], "power key is missing under Replicas"
                assert isinstance(temp[model]['replicas'][replica]['power'], type(st.session_state["config_session"]["densenet121"]['replicas'][1]['power'])) == True, "power key has value of different datatype"
                assert 'throughput' in temp[model]['replicas'][replica], "throughput key is missing under Replicas"
                assert isinstance(temp[model]['replicas'][replica]['throughput'], type(st.session_state["config_session"]["densenet121"]['replicas'][1]['throughput'])) == True, "throughput key has value of different datatype"

        st.session_state["config_session"].update(temp)
        st.success("Uploaded successfully!")
        st.write(temp)
    except AssertionError as warn:
        st.error("Upload Failed: The file schema is invalid.")
        st.error(warn)
