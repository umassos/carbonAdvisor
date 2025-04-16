import pandas as pd
import datetime
import yaml
import os

class CarbonIntensityService:
    def __init__(self, location, start_date, deadline, task, slack=0):
        self.location = location
        self.start_date = start_date
        self.deadline = deadline
        self.slack = slack
        self.task = task
        self.csv_path = f"traces/{self.location}.csv"

    def load_carbon_trace(self):
        csv_path = f"traces/{self.location}.csv"
        carbon_t = pd.read_csv(csv_path)
        carbon_t = carbon_t[["datetime", "carbon_intensity_avg"]]
        if carbon_t.isnull().values.any():
            carbon_t["carbon_intensity_avg"] = carbon_t['carbon_intensity_avg'].interpolate(method='polynomial', order=2)
        carbon_t["carbon_intensity_avg"] /= 1000 * 3600
        start_date = datetime.datetime.strptime(self.start_date, "%Y-%m-%d").date()
        return carbon_t[
            (carbon_t["datetime"] >= self.start_date) & 
            (carbon_t["datetime"] <= str(start_date + datetime.timedelta(hours=self.deadline+self.slack)))
        ]
    
    def load_scale_profile(self):
        with open('./scale_profile.yaml', 'r') as f:
            scale_profile = yaml.safe_load(f)

        tp = scale_profile[self.task]
        return tp["replicas"]

    def get_carbon_intensity(self):
        """Returns the filtered carbon intensity trace."""
        return self.carbon_trace["carbon_intensity_avg"].values / 1000  # Convert to kW

    def get_carbon_at_time(self, time_slot):
        """Returns the carbon intensity for a specific time slot."""
        return self.carbon_trace["carbon_intensity_avg"].iloc[time_slot] / 1000  # Convert to kW
