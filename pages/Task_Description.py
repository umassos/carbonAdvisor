#!/usr/bin/env python3
"""
    Created date: 9/12/22
"""

import yaml
import numpy as np
import streamlit as st
import plotly.graph_objects as go


st.sidebar.markdown("### Task Description")
st.markdown("## Task Description")

profile_path = "scale_profile.yaml"
with open(profile_path, "r") as f:
    profile_data = yaml.safe_load(f)

tasks = {}
for name, task in profile_data.items():
    task_name = f"{task['dataset']}-{name}"

    num_profile = max(task["replicas"])

    tp_table = np.zeros(num_profile + 1)
    energy_table = np.zeros_like(tp_table)

    for num_workers, profile in task["replicas"].items():
        tp_table[num_workers] = profile["throughput"]
        energy_table[num_workers] = profile["gpuPower"]

    tasks[task_name] = {
        "tp_table": tp_table,
        "gpu_power_table": energy_table
    }

ds_size_map = {
    "tinyimagenet": 100000,
    "imagenet": 1281167,
}

selected_tasks = st.sidebar.multiselect("Tasks", tasks.keys(), default=["tinyimagenet-resnet18"])
cpu_power_offset = int(st.sidebar.number_input("CPU Power offset (W)", min_value=0, max_value=100, value=50))

st.markdown("""This page description the profile behavior of different tasks. The performance is profiled on
               AWS `p2.xlarge` instances with Nvidia K80 GPU. 
            """)


st.markdown("### Number of Nodes vs. Throughput")
tp_fig = go.Figure()
for task_name in selected_tasks:
    ds_name = task_name.split('-')[0]
    ds_size = ds_size_map[ds_name]

    tp_per_epoch = tasks[task_name]["tp_table"]/ds_size
    tp_fig.add_trace(
        go.Scatter(y=tp_per_epoch, name=task_name,
                   hovertemplate="# Nodes: %{x}<br>Epochs per hour: %{y:.2f}")
    )

    tasks[task_name]["tp_per_epoch"] = tp_per_epoch

tp_fig.update_yaxes(title_text="Epochs / hour")
tp_fig.update_xaxes(title_text="# Nodes")

st.plotly_chart(tp_fig)


st.markdown("### Number of Nodes vs. Energy Footprint")
energy_fig = go.Figure()
for task_name in selected_tasks:
    num_nodes = np.arange(tasks[task_name]["gpu_power_table"].shape[0])
    power_offset = num_nodes * cpu_power_offset
    energy_footprint = (tasks[task_name]["gpu_power_table"] + power_offset) * 3600 / 3.6e+6

    energy_fig.add_trace(
        go.Scatter(y=energy_footprint, name=task_name,
                   hovertemplate="# Nodes: %{x}<br>Energy per hour: %{y:.2f} KwH")
    )
    tasks[task_name]["energy_footprint"] = energy_footprint

st.plotly_chart(energy_fig)

st.markdown("### Number of Nodes vs. Energy per Throughput")
energy_per_tp_fig = go.Figure()
for task_name in selected_tasks:
    energy_per_tp_fig.add_trace(
        go.Scatter(y=tasks[task_name]["energy_footprint"] / (tasks[task_name]["tp_per_epoch"] + 1e-6) , name=task_name,
                   hovertemplate="# Nodes: %{x}<br>Energy / Throughput: %{y:.2f} KwH/Epoch")
    )

st.plotly_chart(energy_per_tp_fig)
