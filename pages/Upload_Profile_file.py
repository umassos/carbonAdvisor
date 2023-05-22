"""
This represents a page in carbonAdvisor to upload profile or trace files to the system.
It takes in a profile yaml file as input, validates it and stores the information in a session.
The session storage is then later accessed by respective files to get "task_profile".

Author: Swathi Natarajan <swathinatara@umass.edu>
Created: March 15, 2023
Version: 1.0
"""

import streamlit as st
import requests
import yaml
import jsonschema
import json
from Update_Session import fUpdateSessionDefaultProfile

st.markdown("## Upload Profile Files")

#--------------------------------Implemented the profile upload using session state---------------------------------------#

config_session = st.file_uploader("Please upload a .csv file in the same format as listed in the sample")
#Show an example profile.yaml in the UploadFiles
button = st.button("Show sample profile")

if button:
    with open("profile_sample.yaml", "r") as f:
        content = yaml.load(f, Loader=yaml.FullLoader)
        config_session = None
        st.write(content)

#Basic Schema yaml file structure
schema = {
    "type": "object",
    "patternProperties": {
        "^\\w+$": {
            "type": "object",
            "properties": {
                "instance": {"type": "string"},
                "replicas": {
                    "type": "object",
                    "patternProperties": {
                        "^\\d+$": {
                            "type": "object",
                            "properties": {
                                "power": {"type": "number"},
                                "throughput": {"type": "number"}
                            },
                            "required": ["power", "throughput"]
                        }
                    }
                }
            },
            "required": ["instance", "replicas"]
        }
    },
    "additionalProperties": False
}

# Call the library to update the session state "Config_session" if it is not available.
if "config_session" not in st.session_state:
    fUpdateSessionDefaultProfile()

if config_session is not None:
    temp = yaml.safe_load(config_session)
    try:
        #Schema validation using validate() function
        json_content = json.dumps(temp)
        jsonschema.validate(json.loads(json_content), schema)
        
        #Obtaining the instance types from AWS API and using it to validate values of "instance"
        api_url = "https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(Ohio)/Linux/index.json"
        response = requests.get(api_url)
        res = response.json()
        regions = list(res["regions"].keys())
        instances = set()
        for region in regions:
            confs = list(res["regions"][region].keys())
            for conf in confs:
                instances.add(res["regions"][region][conf]["Instance Type"])

        #Custom Validations
        #validation of the new uploaded yaml file:
        for model in temp:
            #1. If the model already exists
            #assert model not in st.session_state["config_session"].keys(), "Keys already found in the existing profile"

            #2. Check if the given instance is among the AWS approved instances
            assert temp[model]["instance"] in instances, "{} is NOT among the AWS approved instances".format(temp[model]["instance"])
            
            #3. check if 'replicas' values are continous.
            assert list(range(min(temp[model]['replicas'].keys()), max(temp[model]['replicas'].keys())+1)) == list(temp[model]['replicas'].keys()), "The replica numbers are not continuous"

        st.session_state["config_session"].update(temp)
        st.success("Uploaded successfully!")
        st.write(temp)
    except AssertionError as warn:
        st.error("Upload Failed: The file schema is invalid.")
        st.error(warn)
    except jsonschema.ValidationError as warn:
        st.error("Upload Failed: The file schema is invalid.")
        st.error(warn)