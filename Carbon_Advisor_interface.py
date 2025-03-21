import os
import pandas as pd
import numpy as np
from glob import glob
import datetime
import streamlit as st
from abc import ABC, abstractmethod
from Update_Session import fUpdateSessionDefaultProfile
 
class CarbonAdvisorMethods(ABC):
    def __init__(self, **kwargs):
        self.algo_inputs = kwargs
        if "config_session" not in st.session_state:
            fUpdateSessionDefaultProfile()
        tp = st.session_state["config_session"][self.algo_inputs['task']]
        self.task_profile = tp["replicas"]
        self.deadline = self.algo_inputs['deadline']
        self.slack = self.algo_inputs['slack'] if 'slack' in self.algo_inputs else 0
        self.task_length = self.algo_inputs['task_length']
        # print("printing task profile 1", self.task_profile)
        carbon_traces_path = sorted(glob("traces/*.csv"))
        carbon_trace_names = [os.path.basename(trace_name) for trace_name in carbon_traces_path]
        carbon_trace_names = [os.path.splitext(trace_name)[0] for trace_name in carbon_trace_names]
        self.carbon_trace_map = {trace_name: trace_path for trace_name, trace_path in zip(carbon_trace_names, carbon_traces_path)}
        carbon_t = pd.read_csv(self.carbon_trace_map[self.algo_inputs['location']])
        start_date = datetime.datetime.strptime(self.algo_inputs['start_date'], "%Y-%m-%d").date()

        self.carbon_trace = carbon_t[
            (carbon_t["datetime"] >= self.algo_inputs['start_date']) & 
            (carbon_t["datetime"] <= str(start_date + datetime.timedelta(hours=self.deadline+self.slack)))
        ]

        print("length of carbon trace", len(self.carbon_trace))

    @abstractmethod
    def compute_schedule(self):
        pass

