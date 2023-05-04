#!/usr/bin/env python3
"""
This represents a page in carbonAdvisor to visualize the carbon consumption with respect to the various completion time.
Input: locational Carbon trace data, batch task, task length, Max workers and time.
Output: carbon consumption vs completion Time

Author: Swathi Natarajan <swathinatara@umass.edu>
Created: March 15, 2023
Version: 1.0
"""

import os
import pandas as pd
import numpy as np
import yaml
import datetime
import streamlit as st

import environment
import agent
import eval_util

import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from dateutil import parser
from glob import glob
from Update_Session import fUpdateSessionDefaultProfile

# Constants

cpu_power_offset = 50

# Loads carbon_trace_map from traces/*
carbon_traces_path = sorted(glob("traces/*.csv"))
carbon_trace_names = [os.path.basename(trace_name) for trace_name in carbon_traces_path]
carbon_trace_names = [os.path.splitext(trace_name)[0] for trace_name in carbon_trace_names]
carbon_trace_map = {trace_name: trace_path for trace_name, trace_path in zip(carbon_trace_names, carbon_traces_path)}

# Call the library to update the session state "Config_session" if it is not available.
if "config_session" not in st.session_state:
    fUpdateSessionDefaultProfile()

# Updates task_profile from the scale_profile.yaml that is in session storage
task_profile = st.session_state["config_session"]

st.sidebar.markdown("### Policy Model")

# Gets the input from user selection
selected_trace = st.sidebar.selectbox("Carbon Trace", options=carbon_trace_names)
carbon_trace = pd.read_csv(carbon_trace_map[selected_trace])
carbon_trace = carbon_trace[carbon_trace["carbon_intensity_avg"].notna()]
carbon_trace["datetime"] = carbon_trace['timestamp'].apply(lambda d: datetime.datetime.fromtimestamp(d))
carbon_trace["date"] = pd.to_datetime(carbon_trace['datetime']).dt.date
carbon_trace["hour"] = pd.to_datetime(carbon_trace['datetime']).dt.hour


selected_task = st.sidebar.selectbox("Task", options=task_profile.keys())
max_workers_allowed = len(task_profile[selected_task]["replicas"])
input_task_length = int(st.sidebar.number_input("Task Length (hour)", min_value=1, value=24))
input_max_workers = int(st.sidebar.number_input("Max Workers", min_value=1, max_value=max_workers_allowed, value=max_workers_allowed))
input_started_date = st.sidebar.date_input("Started Date", min_value=carbon_trace["date"].min(),
                                           max_value=carbon_trace["date"].max(), value=carbon_trace["date"].min())

started_datetime_df = carbon_trace[carbon_trace["date"] == input_started_date]
input_started_hour = int(st.sidebar.number_input("Started Hour", min_value=started_datetime_df["hour"].min(),
                                             max_value=started_datetime_df["hour"].max(),
                                             value=started_datetime_df["hour"].min()))
started_hour_time = datetime.time(hour=input_started_hour)
started_datetime = datetime.datetime.combine(input_started_date, started_hour_time)
started_index = carbon_trace.index[carbon_trace["datetime"] == started_datetime][0]


st.markdown("## Effect of completion time on the carbon footprint")

model_profile = task_profile[selected_task]
num_profile = max(model_profile["replicas"])

tp_table = np.zeros(num_profile+1)
energy_table = np.zeros_like(tp_table)

for num_workers, profile in model_profile["replicas"].items():
    tp_table[num_workers] = profile["throughput"]
    energy_table[num_workers] = profile["power"] + (cpu_power_offset * num_workers)  

energy_table = energy_table * 3600. / 3.6e+6  
num_epochs = tp_table[1] * input_task_length

reward = environment.NonLinearReward(tp_table, energy_table)



# Algorithm:
# 1. Take 6 types of task completion time which is set at 1x, 1.5x, 2x, 2.5x, 3x, 3.5x of the task length.
# 2. For each task completion time,carbon_cost using eval_util.simulate_agent(), 
#     carbon_savings from the carbon_cost and store it in an array
# 3. Plot the line graph with x axis as completionTime and the y axis as carbon_consumption. 

new_tp_fig = go.Figure()
list_of_completion_times = [input_task_length, int(1.5*input_task_length), 2*input_task_length, int(2.5*input_task_length), 3*input_task_length, int(3.5*input_task_length)]
scale_table = np.zeros(6)
carbon_savings = np.zeros(6)

count = 0
arr = np.zeros((6,3))

for len in list_of_completion_times:
    num_epochs_t = tp_table[1] * input_task_length

    reward_t = environment.NonLinearReward(tp_table, energy_table)

    env_t = environment.CarbonOnlyEnvironment(carbon_trace["carbon_intensity_avg"].values,
                                            reward_t, np.array([started_index]), num_epochs_t)
    carbon_scale_agent_t = agent.CarbonScaleAgent(tp_table, energy_table, int(input_max_workers), int(len))
    carbon_cost_scale_t, carbon_scale_states_t, carbon_scale_action_t, exec_time = \
        eval_util.simulate_agent(carbon_scale_agent_t, env_t, int(len))
    carbon_scale_action_t = carbon_scale_action_t.flatten() 

    scale_table[count] = carbon_cost_scale_t[0]
    carbon_savings[count] = (1 - (scale_table[count]/scale_table[0]))*100
    arr[count] = [len, scale_table[count], carbon_savings[count]]
    count = count + 1

df = pd.DataFrame(arr, columns=['length', 'scale', 'savings'])

sched_fig_t = make_subplots(specs=[[{"secondary_y": True}]])
sched_fig_t.add_trace(
    go.Scatter(x=df["length"],
               y=df["scale"],
               hovertext = df["savings"],
               mode="lines+markers+text", 
               hovertemplate="Completion Time: %{x}hrs<br>Carbon Consumption: %{y:.2f}g<br>Carbon Savings: %{hovertext:.2f}%",
               text=round(df["scale"],2),
               textposition='bottom right',
               ),
    secondary_y=False
)

sched_fig_t.update_yaxes( title_text="Carbon Consumption (g)", secondary_y=False, rangemode="tozero" )
sched_fig_t.update_xaxes(title_text="Completion Time(hours)", dtick = input_task_length/2)
st.plotly_chart(sched_fig_t)