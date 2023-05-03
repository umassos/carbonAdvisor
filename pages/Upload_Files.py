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
            
            #2. For a particular task profile, check if 'dataset' exists and has the value same as the type of schema
            assert 'dataset' in temp[model], "dataset key is missing in the uploaded file" 
            assert isinstance(temp[model]['dataset'], type(st.session_state["config_session"]["densenet121"]['dataset'])) == True, "dataset key has value of different datatype"

            #3.For a particular task profile, check if 'replicas' exists and has the same number of replicas as schema.
            assert 'replicas' in temp[model], "replicas key is missing in the uploaded file"
            assert temp[model]['replicas'] is not None, "Replica list is empty. Minimum 1 replicas should be present"
            assert list(range(min(temp[model]['replicas'].keys()), max(temp[model]['replicas'].keys())+1)) == list(temp[model]['replicas'].keys()), "The replica numbers are not continuous"

            #4. For each replica, check if 'gpuPower' and 'throughput' exists and has the value same as the type of schema["densenet121" replica 1 is taken as schema reference] TBD:: This can be improved by introducing a schema file
            for replica in temp[model]['replicas']:
                assert 'gpuPower' in temp[model]['replicas'][replica], "gpuPower key is missing under Replicas"
                assert isinstance(temp[model]['replicas'][replica]['gpuPower'], type(st.session_state["config_session"]["densenet121"]['replicas'][1]['gpuPower'])) == True, "gpuPower key has value of different datatype"
                assert 'throughput' in temp[model]['replicas'][replica], "throughput key is missing under Replicas"
                assert isinstance(temp[model]['replicas'][replica]['throughput'], type(st.session_state["config_session"]["densenet121"]['replicas'][1]['throughput'])) == True, "throughput key has value of different datatype"

        st.session_state["config_session"].update(temp)
        st.success("Uploaded successfully!")
        st.write(temp)
    except AssertionError as warn:
        st.error("Upload Failed: The file schema is invalid.")
        st.error(warn)

#-------------------------------------------Upload trace files (COMMENTED OUT)-------------------------------------------------#

#
# Upload trace files are currently commented out. This can be used for future implementation.
# Two designs are proposed: one with normal trace upload and the other with drop down.
# This may have to be changed as per specification.
#

# st.markdown("###### Design #1")
# trace = st.file_uploader("Upload trace csv files1")
# def upload():
#     st.empty()
#     if trace is None:
#         st.session_state["upload_state"] = "Upload a file first!"
#     else:
#         data_trace = trace.getvalue().decode('utf-8')
#         filename = "./traces/"+trace.name
#         #check if the file exists
#         path = Path(filename)
#         if path.is_file() is False:
#             st.session_state["upload_state"] = "no file name present. Please change the filename"
#             st.error("No file name present. Please change the filename")
#         else:
#             scale_trace_file = open(filename, "w")
#             scale_trace_file.write(data_trace)
#             scale_trace_file.close()
#             st.session_state["upload_state"] = "Saved successfully"
#             st.success("Upload successful.")
#             #success_function(filename)
#     #st.write(st.session_state["upload_state"])
# st.button("Upload file to Sandbox", on_click=upload)


# st.markdown("###### Design #2")

# trace1 = st.file_uploader("Upload trace csv files2")

# def upload1(file_location):
#     if trace1 is None:
#         st.session_state["upload_state1"] = "Upload a file first!"
#     else:
#         data_trace1 = trace1.getvalue().decode('utf-8')
#         scale_trace_file1 = open(file_location, "w")
#         scale_trace_file1.write(data_trace1)
#         scale_trace_file1.close()
#         st.session_state["upload_state1"] = "Saved successfully"
#         st.success("Upload successful.")


# if trace1 is not None:
#     parent_path = pathlib.Path(__file__).parent.parent.resolve()
#     data_path = os.path.join(parent_path, "traces")
#     onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
#     option = st.selectbox('Pick a dataset', onlyfiles)
#     file_location=os.path.join(data_path, option)
#     st.button("Upload file to Sandbox 1", on_click=upload1, args = (file_location,))
