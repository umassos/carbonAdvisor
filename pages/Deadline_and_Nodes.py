import os
import pandas as pd
import numpy as np
import datetime
import streamlit as st

import environment
import agent
import eval_util

import plotly.graph_objects as go
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

st.sidebar.markdown("### Inputs")

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


st.markdown("## Effect of Carbon Consumption with deadline and number of nodes")

# st.markdown("<p style='font-size: 12px;'>This page allows you to analyze the carbon consumption of a task for a range of deadline and number of nodes. It helps you estimate how changing the deadline and number of nodes will impact the carbon consumption\
#              and allows you to make informed decisions to reduce carbon emissions that suit your needs.</p>\n\n"
#             "<p style='font-size: 11px;'>For example:<br> \
#             1. If you have a task that typically takes 24 hours to complete, you can analyze how the carbon consumption varies when completing the task in 24 hours, 36 hours (1.5x), 48 hours (2x), and so on, up to 84 hours (3.5x).<br>\
#             2. Additionally, you can investigate the impact of the number of nodes on carbon consumption. By adjusting the Max Workers, you can simulate scenarios with varying levels of parallelism from 1, 2, 3, and up to the maximum number of workers.</p>"
#             "<p style='font-size: 12px;'>The results of carbon consumption for each combination of deadline and number of nodes will be displayed in a heatmap, allowing you to choose the right deadline and nodes.</p>",
#          unsafe_allow_html=True)
st.markdown("<p style= 'font-size: 12 px;'> This webpage offers a user-friendly tool for analyzing carbon consumption based on task deadlines and the number of nodes. <br><br>\
            By inputting the original task duration, users can explore how carbon consumption change for extended deadlines, ranging from 1.5x to 3.5x the original time. \
            Furthermore, the tool allows users to assess the impact of scalability by adjusting the number of nodes (Max Workers). \
            The results are presented visually in a heatmap, enabling users to make informed decisions on reducing carbon usgae tailored to their needs.",
            unsafe_allow_html = True)


model_profile = task_profile[selected_task]
num_profile = max(model_profile["replicas"])

tp_table = np.zeros(num_profile+1)
energy_table = np.zeros_like(tp_table)

for num_workers, profile in model_profile["replicas"].items():
    tp_table[num_workers] = profile["throughput"]
    energy_table[num_workers] = profile["power"] + (cpu_power_offset * num_workers)  

energy_table = energy_table * 3600 / 3.6e+6
reward = environment.NonLinearReward(tp_table, energy_table)
# Algorithm:
# 1. Take 6 types of task completion time which is set at 1x, 1.5x, 2x, 2.5x, 3x, 3.5x of the task length.
# 2. For each task completion time,carbon_cost using eval_util.simulate_agent(), 
#     carbon_savings from the carbon_cost and store it in an array
# 3. Plot the line graph with x axis as completionTime and the y axis as carbon_consumption. 

list_of_completion_times = [input_task_length, int(1.5*input_task_length), 2*input_task_length, int(2.5*input_task_length), 3*input_task_length, int(3.5*input_task_length)]
list_of_max_workers = list(range(1,input_max_workers+1))

consumption = []
for max_workers in list_of_max_workers:
    templist = []
    for len in list_of_completion_times:
        num_epochs = tp_table[1] * input_task_length
        env = environment.CarbonOnlyEnvironment(carbon_trace["carbon_intensity_avg"].values, reward, np.array([started_index]), num_epochs)
        carbon_scale_agent = agent.CarbonScaleAgent(tp_table, energy_table, max_workers, len)
        carbon_cost_scale, carbon_scale_states, carbon_scale_action, exec_time = \
            eval_util.simulate_agent(carbon_scale_agent, env, int(len))
        templist.append(carbon_cost_scale[0])
    consumption.append(templist)

fig = go.Figure(
    data = go.Heatmap(
        x=list_of_completion_times,
        y=list_of_max_workers,
        z=consumption,
        zmin=0,
        colorscale = 'YlOrRd',
        colorbar = dict(title='CO2 cons.')
    ),
    layout = go.Layout(
        xaxis = dict(
            title = "Deadline (hrs)",
            tickmode = "array",
            tickvals = list_of_completion_times,
            ticktext = list_of_completion_times
        ),
        yaxis = dict(
            title = "# of nodes",
            tickmode = "array",
            tickvals = list_of_max_workers,
            ticktext = list_of_max_workers
        ),
    )
)
st.plotly_chart(fig)
st.markdown("#### For multiple samples")
input_num_samples = int(st.number_input("Sample size", step=1, min_value=1, max_value=50000, value=1000))
list_of_completion_times = [input_task_length, int(1.5*input_task_length), 2*input_task_length, int(2.5*input_task_length), 3*input_task_length, int(3.5*input_task_length)]
list_of_max_workers = list(range(1,input_max_workers+1))

consumption2 = []
hoverlist = []
for max_workers in list_of_max_workers:
    templist2 = []
    hoverlisttemp = []
    for len in list_of_completion_times:
        started_index_batch = np.random.randint(0, carbon_trace.shape[0]-len, input_num_samples)
        if(input_num_samples == 1):
            started_index_batch = np.array([started_index])
        num_epochs = tp_table[1] * input_task_length
        env2 = environment.CarbonOnlyEnvironment(carbon_trace["carbon_intensity_avg"].values, reward, started_index_batch, num_epochs)
        carbon_scale_agent2 = agent.CarbonScaleAgent(tp_table, energy_table, max_workers, len)
        carbon_cost_scale2, carbon_scale_states2, carbon_scale_action2, exec_time2 = \
            eval_util.simulate_agent(carbon_scale_agent2, env2, int(len))   
        templist2.append(np.mean(carbon_cost_scale2))
        hoverlisttemp.append([
            np.min(carbon_cost_scale2), 
            np.max(carbon_cost_scale2),
            np.mean(carbon_cost_scale2),
            np.std(carbon_cost_scale2)
        ])
    consumption2.append(templist2)
    hoverlist.append(hoverlisttemp)



fig2 = go.Figure(
    data = go.Heatmap(
        x=list_of_completion_times,
        y=list_of_max_workers,
        z=consumption2,
        zmin=0,
        colorscale = 'YlOrRd',
        customdata = hoverlist,
        hovertemplate =
            "<br>Carbon Cosumption (g)"
            "<br>Min: %{customdata[0]:.2f}" + 
            "<br>Max: %{customdata[1]:.2f}" +
            "<br>Mean: %{customdata[2]:.2f}" +
            "<br>Std: %{customdata[3]:.2f}<extra></extra>",
    ),
    layout = go.Layout(
        xaxis = dict(
            title = "Deadline (hrs)",
            tickmode = "array",
            tickvals = list_of_completion_times,
            ticktext = list_of_completion_times
        ),
        yaxis = dict(
            title = "# of nodes",
            tickmode = "array",
            tickvals = list_of_max_workers,
            ticktext = list_of_max_workers
        ),
    )
)
st.plotly_chart(fig2)