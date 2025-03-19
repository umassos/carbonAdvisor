import streamlit as st
from Carbon_Advisor_interface import CarbonAdvisorMethods

class CarbonScalerAlgo(CarbonAdvisorMethods):

    def compute_schedule(self):
        print("hello")
        carbon_t = self.carbon_trace["carbon_intensity_avg"].values/1000
        marginal_capacity_carbon = []
        max_nodes = len(self.task_profile)
        print("printing max nodes ", max_nodes)
        for i in range(0, len(self.carbon_trace)):
            for j in range(1, max_nodes + 1):
                value = self.task_profile[j]["throughput"] / \
                    (self.task_profile[j]["power"]
                    * carbon_t[i])
                marginal_capacity_carbon.append(
                    {
                        "time": i,
                        "nodes": j,
                        "value": value
                    }
                )
        # print("printing marginal ", marginal_capacity_carbon)
        marginal_capacity_carbon = sorted(
        marginal_capacity_carbon, key=lambda x: x["value"], reverse=True)

        total_samples = int(self.task_length * self.task_profile[1]["throughput"] * 3600)
        
        task_schedule: dict[int, int] = {}
        done = 0
        while done < total_samples:
            for i in range(0, len(marginal_capacity_carbon)):
                selection = marginal_capacity_carbon[i]
                done += self.task_profile[selection["nodes"]]["throughput"] * 3600
                task_schedule[selection["time"]]= selection["nodes"]
                del marginal_capacity_carbon[i]
                break
        print("task schedule", task_schedule)
        return task_schedule




scaler = CarbonScalerAlgo(deadline=50, slack=8, num_workers=8, task_length=24, location='AU-SA', task='densenet121', start_date='2021-03-22', start_hour=8)
scaler.carbon_scaler_algo()

