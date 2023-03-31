#!/usr/bin/env python3
"""
    Created date: 9/12/22
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
input_task_length = int(st.sidebar.number_input("Task Length (hour)", min_value=1, value=24))
input_deadline = int(st.sidebar.number_input("Deadline", min_value=input_task_length, value=input_task_length))
input_max_workers = int(st.sidebar.number_input("Max Workers", min_value=1, max_value=8, value=8))
input_started_date = st.sidebar.date_input("Started Date", min_value=carbon_trace["date"].min(),
                                           max_value=carbon_trace["date"].max(), value=carbon_trace["date"].min())
started_datetime_df = carbon_trace[carbon_trace["date"] == input_started_date]

input_started_hour = int(st.sidebar.number_input("Started Hour", min_value=started_datetime_df["hour"].min(),
                                             max_value=started_datetime_df["hour"].max(),
                                             value=started_datetime_df["hour"].min()))

started_hour_time = datetime.time(hour=input_started_hour)
started_datetime = datetime.datetime.combine(input_started_date, started_hour_time)

started_index = carbon_trace.index[carbon_trace["datetime"] == started_datetime][0]

st.markdown("## Carbon Footprint Analyzer for ML Tasks")


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

# Carbon scale method
env = environment.CarbonOnlyEnvironment(carbon_trace["carbon_intensity_avg"].values,
                                        reward, np.array([started_index]), num_epochs)
carbon_scale_agent = agent.CarbonScaleAgent(tp_table, energy_table, input_max_workers, input_deadline)
carbon_cost_scale, carbon_scale_states, carbon_scale_action, exec_time = \
    eval_util.simulate_agent(carbon_scale_agent, env, input_deadline)
carbon_scale_action = carbon_scale_action.flatten()

# WaitAWhile method
epochs_per_unit_time = tp_table[1]
num_workers = 1

env = environment.CarbonOnlyEnvironment(carbon_trace["carbon_intensity_avg"].values,
                                        reward, np.array([started_index]), num_epochs)
wait_awhile_agent = agent.WaitAWhileOptimalAgent(epochs_per_unit_time, num_workers, input_deadline)

carbon_cost_waitawhile, wait_awhile_states, wait_awhile_action, exec_time = \
    eval_util.simulate_agent(wait_awhile_agent, env, input_deadline)
wait_awhile_action = wait_awhile_action.flatten()

# Carbon Agnostic method
env = environment.CarbonOnlyEnvironment(carbon_trace["carbon_intensity_avg"].values,
                                        reward, np.array([started_index]), num_epochs)
carbom_agnostic_agent = agent.NaiveAgent(epochs_per_unit_time, num_workers, input_deadline)
carbon_cost_naive, naive_states, naive_action, exec_time = \
    eval_util.simulate_agent(carbom_agnostic_agent, env, input_deadline)
naive_action = naive_action.flatten()

target_period_df = carbon_trace.iloc[started_index:started_index+input_deadline]

sched_fig = make_subplots(specs=[[{"secondary_y": True}]])
sched_fig.add_trace(
    go.Scatter(x=target_period_df["datetime"],
               y=target_period_df["carbon_intensity_avg"],
               mode="lines+markers", name="Carbon Intensity",
               hovertemplate="%{x}<br>%{y:.2f} g/KwH"),
    secondary_y=False
)

visualized_opt = ["CarbonScale", "WaitAWhile", "Carbon Agnostic"]
selected_policy = st.selectbox("Visualize Policy Schedule", visualized_opt)

if selected_policy == "CarbonScale":
    action = carbon_scale_action
elif selected_policy == "WaitAWhile":
    action = wait_awhile_action
elif selected_policy == "Carbon Agnostic":
    action = naive_action

energy_footprint = energy_table[action].round(2)
total_tp = tp_table[action].round(2)
carbon_footprint_t = (energy_footprint * target_period_df["carbon_intensity_avg"]).round(2)

hover_text = []
for e, t, c in zip(energy_footprint, total_tp, carbon_footprint_t):
    hover_text.append(f"Energy: {e:.2f} KwH<br>Throughput: {t:.2f} epochs<br>Carbon: {c:.2f} g")

sched_fig.add_trace(
    go.Bar(x=target_period_df["datetime"], y=action, name="# nodes", hovertext=hover_text,
           hovertemplate="%{x}<br># nodes: %{y}<br>%{hovertext}"),
    secondary_y=True
)


sched_fig.update_traces(secondary_y=True, marker_color="Orange", opacity=0.6)
sched_fig.update_yaxes(range=[0, target_period_df["carbon_intensity_avg"].max()*1.2],
                       title_text="Carbon Intensity (g/KwH)", secondary_y=False)
sched_fig.update_yaxes(range=[0, 10], secondary_y=True, title_text="# nodes")
sched_fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=0.66
))
sched_fig.update_yaxes(secondary_y=True, showgrid=False)

st.plotly_chart(sched_fig)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Carbon Agnostic Footprint", f"{carbon_cost_naive[0]:.2f}g")

with col2:
    carbon_saving = 1 - (carbon_cost_waitawhile / carbon_cost_naive)[0]
    st.metric("WaitAWhile Footprint", f"{carbon_cost_waitawhile[0]:.2f}g", f"saved {carbon_saving*100:.2f}%")

with col3:
    carbon_saving = 1 - (carbon_cost_scale / carbon_cost_naive)[0]
    st.metric("Carbon Scale Footprint", f"{carbon_cost_scale[0]:.2f}g", f"saved {carbon_saving*100:.2f}%")


### Batch sampling
st.markdown("## Batch Sampling")
st.markdown("Launch $$n$$ tasks with random start time")
input_num_samples = int(st.number_input("Sample size", step=1, min_value=1, max_value=50000, value=1000))

started_index_batch = np.random.randint(0, carbon_trace.shape[0]-input_deadline, input_num_samples)

# carbon scale
env = environment.CarbonOnlyEnvironment(carbon_trace["carbon_intensity_avg"].values,
                                        reward, started_index_batch, num_epochs)
carbon_cost_scale_batch, carbon_scale_states_batch, carbon_scale_action_batch, exec_time = \
    eval_util.simulate_agent(carbon_scale_agent, env, input_deadline)

# wait awhile
env = environment.CarbonOnlyEnvironment(carbon_trace["carbon_intensity_avg"].values,
                                        reward, started_index_batch, num_epochs)
carbon_cost_waitawhile_batch, wait_awhile_states_batch, wait_awhile_action_batch,exec_time = \
    eval_util.simulate_agent(wait_awhile_agent, env, input_deadline)

# carbon agnostic
env = environment.CarbonOnlyEnvironment(carbon_trace["carbon_intensity_avg"].values,
                                        reward, started_index_batch, num_epochs)
carbon_cost_naive_batch, naive_states_batch, naive_action_batch, exec_time = \
    eval_util.simulate_agent(carbom_agnostic_agent, env, input_deadline)


carbon_scale_cov = np.std(carbon_scale_states_batch.carbon_intensity, axis=1) / \
                   np.mean(carbon_scale_states_batch.carbon_intensity, axis=1)
carbon_scale_saving = (1 - carbon_cost_scale_batch / carbon_cost_naive_batch) * 100
carbon_scale_started_t = carbon_trace.iloc[started_index_batch]["datetime"]

wait_awhile_cov = np.std(wait_awhile_states_batch.carbon_intensity, axis=1) / \
                  np.mean(wait_awhile_states_batch.carbon_intensity, axis=1)
wait_awhile_saving = (1 - carbon_cost_waitawhile_batch / carbon_cost_naive_batch) * 100
wait_awhile_started_t = carbon_trace.iloc[started_index_batch]["datetime"]

batch_fig = go.Figure()
batch_fig.add_trace(
    go.Scatter(x=carbon_scale_cov, y=carbon_scale_saving, mode="markers", name="Carbon Scale",
               hovertext=carbon_scale_started_t, hovertemplate="Start Time: %{hovertext}<br>Carbon Saving: %{y:.2f}%")
)
batch_fig.add_trace(
    go.Scatter(x=wait_awhile_cov, y=wait_awhile_saving, mode="markers", name="WaitAWhile",
               hovertext=wait_awhile_started_t, hovertemplate="Start Time: %{hovertext}<br>Carbon Saving: %{y:.2f}%")
)

batch_fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=0.66
))
batch_fig.update_yaxes(title_text="Carbon Saving % (w.r.t Carbon Agnostic)")
batch_fig.update_xaxes(title_text="Carbon Intensity Coefficient of Variance")
batch_fig.update_traces(opacity=0.8)

st.plotly_chart(batch_fig)
