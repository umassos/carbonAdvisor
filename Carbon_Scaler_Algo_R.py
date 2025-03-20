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


    def get_scale_at_a_time(self, task_schedule):
        # Extract the number of worker nodes assigned at each time step.
        total_work = 0
        # total_samples = int(self.task_length * self.task_profile[1]["throughput"] * 3600)  # job unable to finish. 
        total_samples = int(self.task_length * self.task_profile[1]["throughput"])
    
        print(f"Required Total Work (total_samples): {total_samples}")

        for time, nodes in task_schedule.items():
            work_done = self.task_profile[nodes]["throughput"] * 3600
            total_work += work_done

            print(f"Time step: {time}, No. of workers: {nodes}, Work done: {work_done}, Total amount of work done: {total_work}")

            if total_work >= total_samples:
                print(f"Job completes at time step {time}")
                return time, {t: n for t, n in task_schedule.items()}
        
        print("Job did not complete within given task schedule.")
        return None, {t: n for t, n in task_schedule.items()} 

    def total_emission(self, task_schedule):
        
        return sum(self.carbon_trace.iloc[t]["carbon_intensity_avg"] * self.task_profile[nodes]["power"] for t, nodes in task_schedule.items())

    def total_energy(self, task_schedule):
        return sum(self.task_profile[nodes]["power"] for nodes in task_schedule.values())

    def analyse_schedule(self):
        # Runs all the functions and return results.
       
        task_schedule = self.compute_schedule()
        completion_time, scale_at_time = self.get_scale_at_a_time(task_schedule)
        return {
            "completion_time": completion_time,
            "scale_at_time": scale_at_time,
            "total_emission": self.total_emission(task_schedule),
            "total_energy": self.total_energy(task_schedule),
        }
    
scaler = CarbonScalerAlgo(deadline=50, slack=8, num_workers=8, task_length=24, location='AU-SA', task='densenet121', start_date='2021-03-22', start_hour=8)
results = scaler.analyse_schedule()
# scaler.carbon_scaler_algo()

print(results)