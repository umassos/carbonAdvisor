#!/usr/bin/env python3
"""
    Created date: 9/12/22
"""

import yaml
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from glob import glob
import os
import pandas as pd
from dateutil import parser
import datetime
from plotly.subplots import make_subplots
import environment
import agent
import eval_util

def get_datetime(iso_str):
    dt = parser.parse(iso_str)
    dt = datetime.datetime.combine(dt.date(), datetime.time(hour=dt.hour))
    return dt

# Constants
ds_size_map = {
    "tinyimagenet": 100000,
    "imagenet": 1281167,
}

cpu_power_offset = 50

st.sidebar.markdown("### Inputs")
st.markdown("## Country wise analysis")
st.markdown("### C02 consumption and Prices")

st.markdown("#### 1. Country vs CO2 consumption")
st.markdown("This is a plot which shows countries and their respective CO2 consumption")

profile_path = "scale_profile.yaml"
with open(profile_path, 'r') as f:
    task_profile = yaml.safe_load(f)

carbon_traces_path = sorted(glob("traces/*.csv"))
carbon_trace_names = [os.path.basename(trace_name) for trace_name in carbon_traces_path]
carbon_trace_names = [os.path.splitext(trace_name)[0] for trace_name in carbon_trace_names]
carbon_trace_map = {trace_name: trace_path for trace_name, trace_path in zip(carbon_trace_names, carbon_traces_path)}

carbon_trace_names_test = st.sidebar.multiselect(
    'Countries',carbon_trace_map,default = ["CA-ON"])

carbon_traces = []
min_date_value = []
max_date_value = []

for selected_trace in carbon_trace_names_test:
    carbon_trace = pd.read_csv(carbon_trace_map[selected_trace])

    carbon_trace = carbon_trace[carbon_trace["carbon_intensity_avg"].notna()]
    carbon_trace = carbon_trace[carbon_trace["carbon_intensity_avg"].notna()]
    carbon_trace["hour"] = carbon_trace["zone_datetime"].apply(lambda x: parser.parse(x).hour)
    carbon_trace["datetime"] = carbon_trace["zone_datetime"].apply(get_datetime)
    carbon_trace["date"] = carbon_trace["zone_datetime"].apply(lambda x: parser.parse(x).date())
    carbon_traces.append(carbon_trace)
    min_date_value.append(carbon_trace["date"].min())
    max_date_value.append(carbon_trace["date"].max())




selected_task = st.sidebar.selectbox("Task", options=task_profile.keys())
input_task_length = st.sidebar.number_input("Task Length (hour)", min_value=1, value=24)
input_deadline = st.sidebar.number_input("Deadline", min_value=input_task_length, value=input_task_length)
input_max_workers = st.sidebar.number_input("Max Workers", min_value=1, max_value=8, value=8)



#allow to choose any 24h period - to do
input_started_date = st.sidebar.date_input("Started Date", min_value=max(min_date_value),
                                           max_value=min(max_date_value), value=max(min_date_value))

#need to figure out - to do

min_hour_value = []
max_hour_value = []
for carbon_trace in carbon_traces:
    started_datetime_df = carbon_trace[carbon_trace["date"] == input_started_date]
    min_hour_value.append(started_datetime_df["hour"].min())
    max_hour_value.append(started_datetime_df["hour"].max())

input_started_hour = st.sidebar.number_input("Started Hour", min_value=max(min_hour_value),
                                             max_value=min(max_hour_value),
                                             value=max(min_hour_value))

started_hour_time = datetime.time(hour=input_started_hour)
started_datetime = datetime.datetime.combine(input_started_date, started_hour_time)




sched_fig = make_subplots(specs=[[{"secondary_y": True}]])
# sched_fig1 = make_subplots(specs=[[{"secondary_y": True}]])
carbon_consumption = []
prices = []

model_profile = task_profile[selected_task]
dataset = model_profile["dataset"]
ds_size = ds_size_map[dataset]
num_profile = max(model_profile["replicas"])

tp_table = np.zeros(num_profile+1)
energy_table = np.zeros_like(tp_table)

for num_workers, profile in model_profile["replicas"].items():
    tp_table[num_workers] = profile["throughput"]
    energy_table[num_workers] = profile["gpuPower"] + (cpu_power_offset * num_workers)  # Add CPU energy offset


tp_table = tp_table / ds_size  # to epochs per hour

energy_table = energy_table * 3600. / 3.6e+6   # to Kwh per hour
num_epochs = tp_table[1] * input_task_length

reward = environment.NonLinearReward(tp_table, energy_table)

for carbon_trace in carbon_traces:
    # Carbon scale method
    started_index = carbon_trace.index[carbon_trace["datetime"] == started_datetime][0]
    env = environment.CarbonOnlyEnvironment(carbon_trace["carbon_intensity_avg"].values,
                                            reward, np.array([started_index]), num_epochs)
    carbon_scale_agent = agent.CarbonScaleAgent(tp_table, energy_table, input_max_workers, input_deadline)
    carbon_cost_scale, carbon_scale_states, carbon_scale_action, exec_time = \
        eval_util.simulate_agent(carbon_scale_agent, env, input_deadline)
    # print(carbon_scale_states)
    # st.metric("Carbon Scale Footprint", f"{carbon_cost_scale[0]:.2f}g")
    #to kgs

    #calculate compute time (multiply exec_time * nodes(servers))
    exec_time = np.array(exec_time)
    nodes = np.array(carbon_scale_action.flatten())
    # print(nodes)
    # print(exec_time)
    compute_time = np.sum(np.multiply(nodes,exec_time))
    #need to multiply with instance later - to do
    prices.append(compute_time)
    carbon_consumption.append(carbon_cost_scale[0]/1000)


sched_fig.add_trace(
    go.Bar(x=carbon_trace_names_test,
               y=carbon_consumption,
                name="Carbon Intensity"),
    secondary_y=False
)

# sched_fig1.add_trace(
#     go.Bar(x=carbon_trace_names_test, y=prices,  name="Price per instance"),
#     secondary_y=True
# )



sched_fig.update_yaxes(title_text="Carbon Consumption (Kg)", secondary_y=False )
# sched_fig1.update_yaxes(title_text="Price per instance")
sched_fig.update_xaxes(title_text="Countries",categoryorder='total ascending')
# sched_fig1.update_xaxes(title_text="Countries")
st.plotly_chart(sched_fig)
# st.plotly_chart(sched_fig1)


st.markdown("#### 2. Country and their price/instance")
st.markdown("Note: The Price calculated is the sum of time required to execute multiplies by number of used nodes(servers) at every hour")
df = pd.DataFrame({'Countries': carbon_trace_names_test, 'Prices': list(prices)}, columns=['Countries', 'Prices'])

st.table(df)

st.markdown("The decision of choosing a country would be based on decreasing carbon consumption and also which has relatively lower cost")
