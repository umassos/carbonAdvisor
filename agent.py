#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================================================
# Author: qianlinliang
# Created date: 1/5/22
# Description: 
# =========================================================================================================

import numpy as np

from abc import ABC, abstractmethod
from environment import CarbonState

from typing import Dict


class Agent(ABC):
    """ Agent to control elastic training tasks """

    @abstractmethod
    def get_action(self, states):
        return NotImplemented


class NaiveAgent(Agent):
    """
    An agent that always run in all situations
    """

    def __init__(self, epochs_per_unit_time: float, num_workers: int, deadline: int):
        """ Initialize agent

        Args:
            epochs_per_unit_time: execution throughput in terms of epochs per unit time
            num_workers: number of workers when running
            deadline: deadline of the tasks
        """
        super(NaiveAgent, self).__init__()

        self._epochs_per_unit_time = epochs_per_unit_time
        self._num_workers = num_workers
        self._deadline = deadline

    def get_action(self, states: CarbonState):
        """ Return the actions in terms of number of workers to run

        This agent always return _num_workers for each time t.

        Args:
            states: CarbonState with carbon intensity values in shape (N, t) where N is the number of tasks and t is
            the deadline

        Returns:
            action: ndarray in shape (N, t), each value indicates the number of workers to use for time t

        """
        n, t = states.carbon_intensity.shape
        remain_epochs = states.remain_epochs.copy()
        k = np.ceil(remain_epochs[0] / (self._num_workers * self._epochs_per_unit_time))
        k = int(k)

        assert np.all(k <= self._deadline)

        action = np.zeros([n, t], dtype=int)
        action[:, :k] = self._num_workers

        return action, None
        

class WaitAWhileOptimalAgent(Agent):
    """
    Wait AWhile agent that greedily select the minimal carbon intensity period to run
    """

    def __init__(self, epochs_per_unit_time: float, num_workers: int, deadline: int):
        """ Initialize agent

        Args:
            epochs_per_unit_time: execution throughput in terms of epochs per unit time
            num_workers: number of workers when running
            deadline: deadline of the tasks
        """
        super(WaitAWhileOptimalAgent, self).__init__()

        self._epochs_per_unit_time = epochs_per_unit_time
        self._num_workers = num_workers
        self._deadline = deadline

    def get_action(self, states: CarbonState):
        """ Return the actions in terms of number of workers to run

        This agent takes t carbon intensity values for each tasks, where t is the maximal running period time steps
        for all tasks (i.e. deadline). Then it return t actions for each time step t in terms of number of workers
        to use.

        Args:
            states: CarbonState with carbon intensity values in shape (N, t) where N is the number of tasks and t is
            the deadline

        Returns:
            action: ndarray in shape (N, t), each value indicates the number of workers to use for time t
        """
        carbon_intensity, remain_epochs = states.carbon_intensity, states.remain_epochs.copy()

        n, t = carbon_intensity.shape
        k = np.ceil(remain_epochs[0] / (self._num_workers * self._epochs_per_unit_time))
        k = int(k)

        assert np.all(k <= self._deadline)

        t_idx = np.argsort(carbon_intensity)
        t_idx = t_idx[:, :k]

        action = np.zeros([n, t], dtype=int)
        np.put_along_axis(action, t_idx, self._num_workers, axis=1)

        return action, None


class CarbonScaleAgent(Agent):
    """ Carbon scale agent that choose the best number of workers to run """

    def __init__(self, throughput_table: np.ndarray, energy_table: np.ndarray, max_worker: int, deadline: int):
        """ Initialize agent

        Args:
            throughput_table: i element is the epochs per unit time using i workers
            energy_table: i element is the energy per unit time using i workers
            max_worker: maximum number of workers to use
            deadline: job deadline
        """
        super(CarbonScaleAgent, self).__init__()

        assert len(throughput_table) == len(energy_table)
        
        self.throughput_table = throughput_table[:max_worker+1]
        self.energy_table = energy_table[:max_worker+1]
        self.max_worker = max_worker
        self.deadline = deadline

        n = len(self.throughput_table)
        if n <= self.max_worker + 1:
            pad_size = self.max_worker - n + 2

            self.throughput_table = np.concatenate([self.throughput_table, np.ones(pad_size)*1e-6])
            self.energy_table = np.concatenate([self.energy_table, np.ones(pad_size)*np.inf])


    def get_action(self, states: CarbonState):
        """ Return the actions in terms of number of workers to run

        Args:
            states: CarbonState with carbon intensity values in shape (N, t) where N is the number of tasks and t is
            the deadline

        Returns:
            num_workers: ndarray in shape (N, t), each value indicates the number of workers to use for time t
            exec_time: ndarray in shape (N, t), each value indicate the active fraction of this time unit

        """
        carbon_intensity, remain_epochs = states.carbon_intensity, states.remain_epochs.copy()
        num_workers = np.zeros_like(carbon_intensity, dtype=int)    # shape [n, t]

        carbon_per_tp = self.energy_table[1] * carbon_intensity / self.throughput_table[1]

        while not np.all(remain_epochs == 0):
            running_idx = np.where(remain_epochs > 0)[0]
            min_idx = np.argmin(carbon_per_tp, axis=1)
            t_idx = min_idx[running_idx]

            num_workers[running_idx, t_idx] += 1

            processed_epochs = self.throughput_table[num_workers[running_idx, t_idx]] - \
                self.throughput_table[num_workers[running_idx, t_idx]-1]

            remain_epochs[running_idx] = np.where(remain_epochs[running_idx] > processed_epochs,
                                                  remain_epochs[running_idx] - processed_epochs,
                                                  0)

            # update current state
            cur_num_workers = num_workers[running_idx, t_idx]
            carbon_diff = (self.energy_table[cur_num_workers+1] - self.energy_table[cur_num_workers]) \
                * carbon_intensity[running_idx, t_idx]
            tp_diff = self.throughput_table[cur_num_workers+1] - self.throughput_table[cur_num_workers]
            tp_diff = np.where(tp_diff < 0, 1e-6, tp_diff)

            carbon_per_tp[running_idx, t_idx] = carbon_diff / tp_diff

        # Compute execution time
        exec_time = np.zeros_like(num_workers, dtype=float)
        exec_time[num_workers > 0] = 1

        processed_epochs = np.cumsum(self.throughput_table[num_workers], axis=1)
        over_processed = np.max(processed_epochs, axis=1) - states.remain_epochs

        carbon_per_tp = np.zeros_like(num_workers, dtype=float)
        active_idx = num_workers > 0
        carbon_per_tp[active_idx] = \
            self.energy_table[num_workers[active_idx]] * carbon_intensity[active_idx] / \
            self.throughput_table[num_workers[active_idx]]

        max_idx = np.argmax(carbon_per_tp, axis=1)
        all_idx = np.arange(num_workers.shape[0])

        time_offset = over_processed / self.throughput_table[num_workers[all_idx, max_idx]]
        exec_time[all_idx, max_idx] -= time_offset

        return num_workers, exec_time




