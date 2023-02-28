#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================================================
# Author: qianlinliang
# Created date: 1/4/22
# Description: 
# =========================================================================================================

import numpy as np

from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Tuple, Dict

CarbonState = namedtuple("CarbonState", ["carbon_intensity", "remain_epochs"])


class Environment(ABC):
    """ Abstract class for training environment """

    @abstractmethod
    def take_action(self, action, exec_time):
        return NotImplemented

    @abstractmethod
    def is_end(self):
        return NotImplemented

    @abstractmethod
    def peek(self, t):
        return NotImplemented


class Reward(ABC):
    """ Abstract class for reward """

    @abstractmethod
    def __call__(self, remain_epochs, carbon_intensity, num_workers, exec_time):
        return NotImplemented


class LinearReward(Reward):
    """ Linear scale energy consumption and throughput """
    def __init__(self, epochs_per_unit_time: float, energy_per_unit_time: float):
        """ Reward function assume energy consumption and throughput scale linearly

        Args:
            epochs_per_unit_time: number of epochs per unit time per worker
            energy_per_unit_time: energy per unit time per worker in terms of KwH
        """
        super(LinearReward, self).__init__()

        self._epochs_per_unit_time = epochs_per_unit_time
        self._energy_per_unit_time = energy_per_unit_time

    def __call__(self, remain_epochs: np.ndarray,
                 carbon_intensity: np.ndarray,
                 num_workers: np.ndarray,
                 execution_time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Compute the carbon emission and number of epochs process

        Args:
            remain_epochs: ndarray in shape (N, ) remaining epochs to be processed
            carbon_intensity: ndarray in shape (N, t), average carbon intensity in terms of gCO2/Kwh
            num_workers: ndarray in shape (N, t). Number of workers used for each t time step
            execution_time: ndarray in shape (N, t), each value indicate the active fraction of this time unit

        Returns:
            carbon_emission: ndarray in shape (N, ). Carbon emission with same shape as carbon intensity
            num_epochs_process: ndarray in shape (N, ). Number of epochs process with same shape as carbon intensity

        """

        remain_epochs = remain_epochs.reshape(-1, 1)

        num_epochs_process = self._epochs_per_unit_time * num_workers
        num_epochs_process = np.cumsum(num_epochs_process, axis=1)
        num_epochs_process = np.where(num_epochs_process > remain_epochs,
                                      remain_epochs,
                                      num_epochs_process)

        num_epochs_process_prev = np.zeros_like(num_epochs_process)
        num_epochs_process_prev[:, 1:] = num_epochs_process[:, :-1]

        num_epochs_process = num_epochs_process - num_epochs_process_prev

        running_idx = num_workers > 0

        if execution_time is None:
            execution_time = np.zeros_like(num_workers)
            execution_time[running_idx] = num_epochs_process[running_idx] / \
                (self._epochs_per_unit_time * num_workers[running_idx])

        carbon_emission = self._energy_per_unit_time * num_workers * execution_time * carbon_intensity

        num_epochs_process = np.sum(num_epochs_process, axis=1)
        carbon_emission = np.sum(carbon_emission, axis=1)

        return carbon_emission, num_epochs_process


class NonLinearReward(Reward):
    """ Non-linear scalability for both throughput and energy """
    def __init__(self, throughput_table: np.ndarray, energy_table: np.ndarray):
        """ initialize reward

        Args:
            throughput_table: i element is the epochs per unit time using i workers
            energy_table: i element is the energy per unit time using i workers
        """
        super(NonLinearReward, self).__init__()

        self.throughput_table = throughput_table
        self.energy_table = energy_table

    def __call__(self, remain_epochs: np.ndarray,
                 carbon_intensity: np.ndarray,
                 num_workers: np.ndarray,
                 execution_time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Compute the carbon emission and number of epochs process

        Args:
            remain_epochs: ndarray in shape (N, ) remaining epochs to be processed
            carbon_intensity: ndarray in shape (N, t), average carbon intensity in terms of gCO2/Kwh
            num_workers: ndarray in shape (N, t). Number of workers used for each t time step
            execution_time: ndarray in shape (N, t), each value indicate the active fraction of this time unit

        Returns:
            carbon_emission: ndarray in shape (N, ). Carbon emission with same shape as carbon intensity
            num_epochs_process: ndarray in shape (N, ). Number of epochs process with same shape as carbon intensity

        """
        remain_epochs = remain_epochs.reshape(-1, 1)

        num_epochs_process = self.throughput_table[num_workers] * num_workers
        num_epochs_process = np.cumsum(num_epochs_process, axis=1)
        num_epochs_process = np.where(num_epochs_process > remain_epochs,
                                      remain_epochs,
                                      num_epochs_process)

        num_epochs_process_prev = np.zeros_like(num_epochs_process)
        num_epochs_process_prev[:, 1:] = num_epochs_process[:, :-1]

        num_epochs_process = num_epochs_process - num_epochs_process_prev

        running_idx = num_workers > 0

        if execution_time is None:
            execution_time = np.zeros_like(num_workers, dtype=float)
            execution_time[running_idx] = num_epochs_process[running_idx] / \
                (self.throughput_table[num_workers[running_idx]])

        carbon_emission = self.energy_table[num_workers] * carbon_intensity * execution_time

        num_epochs_process = np.sum(num_epochs_process, axis=1)
        carbon_emission = np.sum(carbon_emission, axis=1)

        return carbon_emission, num_epochs_process


class CarbonOnlyEnvironment(Environment):
    """ Univariate carbon environment """

    def __init__(self, carbon_intensity: np.ndarray, reward_fn: Reward, start_idx: np.ndarray, num_epochs: int):
        """ Initialize environment

        Args:
            carbon_intensity: an array of average carbon intensity
            reward_fn: a reward function to compute the carbon emission and task progress
            start_idx: start index of each task
            num_epochs: number of epochs to train
        """
        super(CarbonOnlyEnvironment, self).__init__()

        self._carbon_intensity = carbon_intensity
        self._reward_fn = reward_fn
        self._start_idx = np.array(start_idx)
        self._cur_idx = np.array(start_idx)
        self._remain_epochs = np.ones_like(self._start_idx, dtype=np.float) * num_epochs

    def take_action(self, action: np.ndarray, exec_time: np.ndarray) -> Tuple[np.ndarray, CarbonState]:
        """ Take action and return reward and next state

        Args:
            action: an integer array in shape (N, t). Indicates the number of workers currently run for each time t
            exec_time: ndarray in shape (N, t), each value indicate the active fraction of this time unit

        Returns:
            reward: reward in terms of negative carbon emission
            next_state: next state

        """

        n, t = action.shape

        assert np.all(self._cur_idx + t < self._carbon_intensity.shape[0])

        carbon_intensity_idx = np.empty_like(action, dtype=np.int)
        for i in range(n):
            carbon_intensity_idx[i, :] = np.arange(self._cur_idx[i], self._cur_idx[i]+t)

        carbon_intensity = self._carbon_intensity[carbon_intensity_idx]
        carbon_emission, num_epochs_process = \
            self._reward_fn(self._remain_epochs, carbon_intensity, action, exec_time)

        self._remain_epochs -= num_epochs_process
        self._cur_idx += t

        reward = -carbon_emission
        next_state = CarbonState(self._carbon_intensity[self._cur_idx].reshape(-1, 1), self._remain_epochs)

        return reward, next_state

    def is_end(self) -> bool:
        """ Check if all tasks are finished

        Returns:
            is_end: True if all tasks are finished, otherwise False

        """
        return np.all(self._remain_epochs == 0)

    def peek(self, t: int) -> CarbonState:
        """ Get the next t states without taking actions

        Args:
            t: number of time steps to peek

        Returns:
            states: next t states

        """
        n = self._cur_idx.shape[0]

        assert np.all(self._cur_idx + t < self._carbon_intensity.shape[0])
        carbon_intensity_idx = np.empty([n, t], dtype=np.int)

        for i in range(n):
            carbon_intensity_idx[i, :] = np.arange(self._cur_idx[i], self._cur_idx[i]+t)

        carbon_intensity = self._carbon_intensity[carbon_intensity_idx]
        return CarbonState(carbon_intensity, self._remain_epochs)


