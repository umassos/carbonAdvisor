#!/usr/bin/env python3
"""
    Created date: 9/6/22
"""

import numpy as np

import environment
import agent


def simulate_agent(target_agent, env, deadline):
    states = env.peek(deadline)
    action, exec_time = target_agent.get_action(states)
    carbon_cost, next_state = env.take_action(action, exec_time)

    assert np.all(np.isclose(next_state.remain_epochs, 0))

    return -carbon_cost, states, action


def compute_carbon_cost_wait_awhile(carbon_trace, num_tasks, num_workers, deadline, start_idx,
                                    epochs_per_unit_time, energy_per_unit_time, num_epochs):
    reward = environment.LinearReward(epochs_per_unit_time, energy_per_unit_time)
    env = environment.CarbonOnlyEnvironment(carbon_trace, reward, start_idx, num_epochs)
    wait_awhile_agent = agent.WaitAWhileOptimalAgent(epochs_per_unit_time, num_workers, deadline)

    carbon_cost, _, _ = simulate_agent(wait_awhile_agent, env, deadline)

    return carbon_cost


def compute_carbon_cost_naive(carbon_trace, num_tasks, num_workers, deadline, start_idx,
                              epochs_per_unit_time, energy_per_unit_time, num_epochs):
    reward = environment.LinearReward(epochs_per_unit_time, energy_per_unit_time)
    env = environment.CarbonOnlyEnvironment(carbon_trace, reward, start_idx, num_epochs)
    naive_agent = agent.NaiveAgent(epochs_per_unit_time, num_workers, deadline)

    carbon_cost, _, _ = simulate_agent(naive_agent, env, deadline)

    return carbon_cost


def compute_carbon_cost_carbon_scale(carbon_trace, deadline, start_idx,
                                     tp_table, energy_table, num_epochs, max_worker):

    reward = environment.NonLinearReward(tp_table, energy_table)
    env = environment.CarbonOnlyEnvironment(carbon_trace, reward, start_idx, num_epochs)
    carbon_scale_agent = agent.CarbonScaleAgent(tp_table, energy_table, max_worker, deadline)

    carbon_cost, _, _ = simulate_agent(carbon_scale_agent, env, deadline)

    return carbon_cost


def carbon_saving_between_naive_and_wait_a_while(carbon_trace, num_tasks, num_workers, deadline, start_idx,
                                                 epochs_per_unit_time, energy_per_unit_time, num_epochs):
    wait_awhile_carbon = compute_carbon_cost_wait_awhile(carbon_trace, num_tasks, num_workers, deadline, start_idx,
                                                         epochs_per_unit_time, energy_per_unit_time, num_epochs)

    naive_carbon = compute_carbon_cost_naive(carbon_trace, num_tasks, 4, deadline, start_idx,
                                             epochs_per_unit_time, energy_per_unit_time, num_epochs)

    return 1 - (wait_awhile_carbon / naive_carbon)

