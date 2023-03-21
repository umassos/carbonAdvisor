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


carbon_traces_path = sorted(glob("traces/*.csv"))
carbon_trace_names = [os.path.basename(trace_name) for trace_name in carbon_traces_path]
carbon_trace_names = [os.path.splitext(trace_name)[0] for trace_name in carbon_trace_names]
carbon_trace_map = {trace_name: trace_path for trace_name, trace_path in zip(carbon_trace_names, carbon_traces_path)}

profile_path = "scale_profile.yaml"
with open(profile_path, 'r') as f:
    task_profile = yaml.safe_load(f)

st.sidebar.markdown("### Policy Model")

selected_trace = st.sidebar.selectbox("Carbon Trace", options=carbon_trace_names)
carbon_trace = pd.read_csv(carbon_trace_map[selected_trace])
carbon_trace = carbon_trace[carbon_trace["carbon_intensity_avg"].notna()]
carbon_trace["hour"] = carbon_trace["zone_datetime"].apply(lambda x: parser.parse(x).hour)
carbon_trace["datetime"] = carbon_trace["zone_datetime"].apply(get_datetime)
carbon_trace["date"] = carbon_trace["zone_datetime"].apply(lambda x: parser.parse(x).date())


selected_task = st.sidebar.selectbox("Task", options=task_profile.keys())
input_task_length = st.sidebar.number_input("Task Length (hour)", min_value=1, value=24)
input_deadline = st.sidebar.number_input("Deadline", min_value=input_task_length, value=input_task_length)
# input_max_workers = st.sidebar.number_input("Max Workers", min_value=1, max_value=8, value=8)

input_started_date = st.sidebar.date_input("Started Date", min_value=carbon_trace["date"].min(),
                                           max_value=carbon_trace["date"].max(), value=carbon_trace["date"].min())
started_datetime_df = carbon_trace[carbon_trace["date"] == input_started_date]

input_started_hour = st.sidebar.number_input("Started Hour", min_value=started_datetime_df["hour"].min(),
                                             max_value=started_datetime_df["hour"].max(),
                                             value=started_datetime_df["hour"].min())

started_hour_time = datetime.time(hour=int(input_started_hour))

started_datetime = datetime.datetime.combine(input_started_date, started_hour_time)

started_index = carbon_trace.index[carbon_trace["datetime"] == started_datetime][0]

st.markdown("## Analysis as per the nodes")


# simulation
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

sched_fig = make_subplots(specs=[[{"secondary_y": True}]])
sched_fig1 = make_subplots(specs=[[{"secondary_y": True}]])
carbon_consumption = []
nodes = []
prices = []
priceOverhead = []

baseCost = 0.0



# Carbon scale method
for input_max_workers in range(2,9):
    epochs_per_unit_time = tp_table[1]
    num_workers = 1
    env = environment.CarbonOnlyEnvironment(carbon_trace["carbon_intensity_avg"].values,
                                        reward, np.array([started_index]), num_epochs)
    carbom_agnostic_agent = agent.NaiveAgent(epochs_per_unit_time, num_workers , input_deadline)
    carbon_cost_naive, naive_states, naive_action, exec_time = \
        eval_util.simulate_agent(carbom_agnostic_agent, env, input_deadline)
    naive_action = naive_action.flatten()

    
    env = environment.CarbonOnlyEnvironment(carbon_trace["carbon_intensity_avg"].values,
                                            reward, np.array([started_index]), num_epochs)
    carbon_scale_agent = agent.CarbonScaleAgent(tp_table, energy_table, input_max_workers, input_deadline)
    carbon_cost_scale, carbon_scale_states, carbon_scale_action, exec_time = \
        eval_util.simulate_agent(carbon_scale_agent, env, int(input_deadline))
    carbon_scale_action = carbon_scale_action.flatten()

    
    carbon_saving = 1 - (carbon_cost_scale / carbon_cost_naive)[0]
    
    nodes.append(input_max_workers)
    
    compute_time = np.sum(np.multiply(carbon_scale_action,exec_time))
    prices.append(compute_time)
    if(input_max_workers == 2):
        baseCost = compute_time
    priceOverhead.append("Price Overhead:" + str(round((((compute_time/baseCost) - 1) * 100),2))+ "% \n" + 
        "Carbon Saving: " +str({carbon_saving*100}) + "%")
    carbon_consumption.append(carbon_cost_scale[0]/1000)




sched_fig.add_trace(
    go.Bar(x=nodes,
               y=carbon_consumption, 
                name="Carbon Intensity"),
    secondary_y=False
)



sched_fig1.add_trace(
    go.Bar(x=nodes,
               y=prices,
               
                name="Price", hovertext=priceOverhead,
           hovertemplate="Price:%{y}<br>%{hovertext}"),
    secondary_y=False
)


sched_fig.update_yaxes(title_text="Carbon Consumption (Kg)", secondary_y=False)
sched_fig1.update_yaxes(title_text="Prices")
sched_fig.update_xaxes(title_text="# Nodes")
sched_fig1.update_xaxes(title_text="# Nodes")

st.markdown("##### 1. Carbon consumption vs Nodes")
st.plotly_chart(sched_fig)

st.markdown("##### 2. Prices vs Nodes")
st.plotly_chart(sched_fig1)