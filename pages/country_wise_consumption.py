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
from Update_Session import fUpdateSessionDefaultProfile
import plotly.express as px
import instance

# Constants

cpu_power_offset = 50

st.sidebar.markdown("### Inputs")
st.markdown("## Country wise analysis")
st.markdown("### Q. Where should you run your job?")
st.markdown("Optimal Country Selection: This approach focuses on identifying the most suitable country based on \
            two key criteria - decreasing carbon consumption and relative lower cost. By considering both environmental \
            sustainability and economic efficiency, decision-makers can prioritize countries that demonstrate a commitment \
             to reducing their carbon footprint while offering cost advantages. This balanced approach ensures that the chosen \
            country aligns with sustainability goals and provides cost-effective solutions, contributing to a greener future.")



st.markdown("#### 1. Carbon consumed to run the job")
st.markdown("This plot showcases the carbon consumption of different countries and provides insights into their relative levels of carbon emissions. \
            The carbon consumption values have been calculated using the Carbon Scaler algorithm, which takes into account various factors to estimate \
            the carbon footprint of each country. By visualizing this data, we can gain a better understanding of the varying carbon consumption patterns \
             across different regions and identify potential areas for environmental improvement and sustainable practices.")

# Call the library to update the session state "Config_session" if it is not available.
if "config_session" not in st.session_state:
    fUpdateSessionDefaultProfile()

# Updates task_profile from the scale_profile.yaml that is in session storage
task_profile = st.session_state["config_session"]

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
    carbon_trace["datetime"] = carbon_trace['timestamp'].apply(lambda d: datetime.datetime.fromtimestamp(d))
    carbon_trace["date"] = pd.to_datetime(carbon_trace['datetime']).dt.date
    carbon_trace["hour"] = pd.to_datetime(carbon_trace['datetime']).dt.hour
    carbon_trace["Country"] = selected_trace
    
    carbon_traces.append(carbon_trace)
    min_date_value.append(carbon_trace["date"].min())
    max_date_value.append(carbon_trace["date"].max())




selected_task = st.sidebar.selectbox("Task", options=task_profile.keys())
input_task_length = int(st.sidebar.number_input("Task Length (hour)", min_value=1, value=24))
input_deadline = int(st.sidebar.number_input("Deadline", min_value=input_task_length, value=input_task_length))
input_max_workers = int(st.sidebar.number_input("Max Workers", min_value=1, max_value=8, value=8))



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

input_started_hour = int(st.sidebar.number_input("Started Hour", min_value=max(min_hour_value),
                                             max_value=min(max_hour_value),
                                             value=max(min_hour_value)))

started_hour_time = datetime.time(hour=input_started_hour)
started_datetime = datetime.datetime.combine(input_started_date, started_hour_time)




sched_fig = make_subplots(specs=[[{"secondary_y": True}]])
# sched_fig1 = make_subplots(specs=[[{"secondary_y": True}]])
countries = []
carbon_consumption = []
prices = []

model_profile = task_profile[selected_task]
min_profile_replicas = min(model_profile["replicas"])
num_profile = max(model_profile["replicas"])

tp_table = np.zeros(num_profile+1)
energy_table = np.zeros_like(tp_table)

for num_workers, profile in model_profile["replicas"].items():
    tp_table[num_workers] = profile["throughput"]
    energy_table[num_workers] = profile["power"] + (cpu_power_offset * num_workers)  # Add CPU energy offset


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

    #calculate compute time (multiply exec_time * nodes(servers))
    exec_time = np.array(exec_time)
    nodes = np.array(carbon_scale_action.flatten())
    # print(np.shape(nodes))
    # print(np.shape(exec_time))
    compute_time = float(np.sum(np.multiply(nodes,exec_time)))
    #need to multiply with instance later - to do
    instance_price = float(instance.get_instance_price(str(model_profile["instance"])))
    
    prices.append(compute_time*instance_price)
    # print(carbon_trace["Country"][0])
    countries.append(carbon_trace["Country"][0])
    
    carbon_consumption.append(carbon_cost_scale[0]/1000)


sched_fig = make_subplots(specs=[[{"secondary_y": True}]])

sched_fig.add_trace(
    go.Bar(x=countries,
               y=carbon_consumption, 
                name="Carbon Consumption in regions"),
    secondary_y=False
)
sched_fig.update_yaxes(title_text="Carbon Consumption (Kg)")
sched_fig.update_xaxes(title_text="Countries")
sched_fig.update_layout(title="A Comparison of Carbon Consumption Across Countries")
st.plotly_chart(sched_fig)


st.markdown("#### 2. Cost incurred to run job")
st.markdown("This plot provides an estimate of the expenses incurred in running an instance  \
            on the Amazon Web Services (AWS) platform. The calculated price takes into account the compute time, which is \
            determined by multiplying the execution time by the number of used nodes or servers for each hour. This compute \
            time is then multiplied by the cost of running an AWS instance specific to the chosen region. By visualizing this \
            data, users can gain insights into the financial implications of running instances on AWS and make informed decisions \
            regarding resource allocation and cost optimization. The plot highlights the importance of considering factors such \
            as execution time, number of nodes, and regional costs when estimating the overall expenses associated with running \
            instances on the AWS platform.")


sched_fig1 = make_subplots(specs=[[{"secondary_y": True}]])

sched_fig1.add_trace(
    go.Bar(x=countries,
               y=prices, 
                name="Prices in regions"),
    secondary_y=False
)

sched_fig1.update_yaxes(title_text="Prices ($)")
sched_fig1.update_xaxes(title_text="Countries")
sched_fig1.update_layout(title="A Comparison of Prices Across Countries")

st.plotly_chart(sched_fig1)


### Batch sampling
st.markdown("#### Batch Sampling")
st.markdown("Selecting a representative subset of data based on user-defined sample size for efficient analysis and insights.")

input_num_samples = int(st.number_input("Sample size", step=1, min_value=1, max_value=50000, value=1000))

countries =[]
carbon_consumption =[]
prices = []
for carbon_trace in carbon_traces:
    # Carbon scale method
    
    started_index_batch = np.random.randint(0, carbon_trace.shape[0]-input_deadline, input_num_samples)
    started_index = carbon_trace.index[carbon_trace["datetime"] == started_datetime][0]
    env = environment.CarbonOnlyEnvironment(carbon_trace["carbon_intensity_avg"].values,
                                            reward, started_index_batch, num_epochs)
    carbon_scale_agent = agent.CarbonScaleAgent(tp_table, energy_table, input_max_workers, input_deadline)
    carbon_cost_scale_batch, carbon_scale_states_batch, carbon_scale_action_batch, exec_time = \
        eval_util.simulate_agent(carbon_scale_agent, env, input_deadline)

    instance_price = float(instance.get_instance_price(str(model_profile["instance"])))
    
    #calculate compute time (multiply exec_time * nodes(servers))
    exec_time = np.array(exec_time)
    nodes = np.array(carbon_scale_action_batch)
    # print(carbon_cost_scale_batch)
    
    result = []

    countries.extend([carbon_trace["Country"][0]]*input_num_samples)
    carbon_consumption.extend(carbon_cost_scale_batch/1000)

    for i in range(input_num_samples):
        flattened_y = exec_time[i].flatten()
        multiplied_values = nodes[i] * exec_time[i]
        prices.append(np.sum(multiplied_values)*instance_price)

    # compute_time = np.sum(np.multiply(nodes,exec_time))
    # #need to multiply with instance later - to do
    # prices.append(compute_time)
    # # print(carbon_trace["Country"][0])
    

st.markdown("#### 1. Carbon consumed to run the job")
        
df = pd.DataFrame({'Countries': countries, 'carbon_consumption': carbon_consumption}, columns=['Countries', 'carbon_consumption'])
# Group the data by country
groups = df.groupby('Countries')

# Create a subplot with 1 row and 3 columns
fig = make_subplots(rows=1, cols=len(set(countries)))

# Loop over each group and create a boxplot for that group
for i, (name, group) in enumerate(groups):
    fig.add_trace(
        go.Box(y=group['carbon_consumption'], name=name),
        row=1, col=1
    )

# Set the layout for the subplots
fig.update_layout(title='Carbon consumption across countries', height=500, width=800)
fig.update_yaxes(title_text="Carbon Consumption (Kg)")
fig.update_xaxes(title_text="Countries")
# Display the subplots
st.plotly_chart(fig)




st.markdown("#### 2. Cost incurred to run the job")
# df = pd.DataFrame({'Countries': carbon_trace_names_test, 'Prices': list(prices)}, columns=['Countries', 'Prices'])

# st.table(df)

df = pd.DataFrame({'Countries': countries, 'prices': prices}, columns=['Countries', 'prices'])

# Group the data by country
groups = df.groupby('Countries')



# Create a subplot with 1 row and 3 columns
fig = make_subplots(rows=1, cols=len(set(countries)))

# Loop over each group and create a boxplot for that group
for i, (name, group) in enumerate(groups):
    
    fig.add_trace(
        
        go.Box(y=group['prices'], name=name),
        row=1, col=1
    )

# Set the layout for the subplots
fig.update_yaxes(title_text="Prices ($)")
fig.update_xaxes(title_text="Countries")
fig.update_layout(title='Cost incurred across countries', height=500, width=800)

# Display the subplots
st.plotly_chart(fig)