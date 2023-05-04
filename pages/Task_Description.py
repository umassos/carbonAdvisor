#!/usr/bin/env python3
"""
    Created date: 9/12/22
"""

import yaml
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from Update_Session import fUpdateSessionDefaultProfile

st.sidebar.markdown("### Task Description")
st.markdown("## Task Description")

# profile_path = "scale_profile.yaml"
# with open(profile_path, "r") as f:
#     profile_data = yaml.safe_load(f)

# Call the library to update the session state "Config_session" if it is not available.
if "config_session" not in st.session_state:
    fUpdateSessionDefaultProfile()

# Updates profile_data from the scale_profile.yaml that is in session storage
profile_data = st.session_state["config_session"]

tasks = {}
for name, task in profile_data.items():
    num_profile = max(task["replicas"])

    tp_table = np.zeros(num_profile + 1)
    energy_table = np.zeros_like(tp_table)
    mc_table = np.zeros_like(tp_table)
    prev_profile = 0
    for num_workers, profile in task["replicas"].items():
        tp_table[num_workers] = profile["throughput"]
        energy_table[num_workers] = profile["power"]
        if num_workers == 0:
            mc_table[num_workers] = 0
        else:
            mc_table[num_workers] = profile["throughput"] - prev_profile
        prev_profile = profile["throughput"]
            

    tasks[name] = {
        "tp_table": tp_table,
        "power_table": energy_table,
        "mc_table": mc_table
    }

selected_tasks = st.sidebar.multiselect("Tasks", tasks.keys(), default=["resnet18"])
cpu_power_offset = int(st.sidebar.number_input("CPU Power offset (W)", min_value=0, max_value=100, value=50))

st.markdown("""This page description the profile behavior of different tasks. The performance is profiled on
               AWS `p2.xlarge` instances with Nvidia K80 GPU. 
            """)


st.markdown("### Number of Nodes vs. Throughput")
tp_fig = go.Figure()
for name in selected_tasks:
    tp = tasks[name]["tp_table"]
    tp_fig.add_trace(
        go.Scatter(y=tp, name=name,
                   hovertemplate="# Nodes: %{x}<br>Samples per hour: %{y:.2f}")
    )

    tasks[name]["tp"] = tp

tp_fig.update_yaxes(title_text="Samples / hour")
tp_fig.update_xaxes(title_text="# Nodes")

st.plotly_chart(tp_fig)


st.markdown("### Marginal Capacity")
marginal_capacity_fig = go.Figure()
for name in selected_tasks:
    marginal_capacity_fig.add_trace(
        go.Scatter(y=tasks[name]["mc_table"], name=name,
                   hovertemplate="# Nodes: %{x} Marginal Capacity %{y:.2f} Samples/hour")
    )

st.plotly_chart(marginal_capacity_fig)


st.markdown("### Number of Nodes vs. Power Consumption")
energy_fig = go.Figure()
for name in selected_tasks:
    num_nodes = np.arange(tasks[name]["power_table"].shape[0])
    power_offset = num_nodes * cpu_power_offset
    power_consumption = (tasks[name]["power_table"] + power_offset)

    energy_fig.add_trace(
        go.Scatter(y=power_consumption, name=name,
                   hovertemplate="# Nodes: %{x}<br>Energy per hour: %{y:.2f} watt")
    )
    tasks[name]["power_consumption"] = power_consumption
    energy_fig.update_yaxes(title_text="Power (watt)")
    energy_fig.update_xaxes(title_text="# Nodes")
    
st.plotly_chart(energy_fig)

