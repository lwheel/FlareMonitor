"""
Oura Ring Data Collection Script for Health ML Model
Fetches historical data from Oura API V2 and saves to CSV files
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class OuraDataCollector:
    """Collect data from Oura Ring API V2"""
    
    def __init__(self, personal_access_token):
        """
        Initialize the Oura API client
        
        Args:
            personal_access_token: Your Oura Personal Access Token
        """
        self.base_url = "https://api.ouraring.com/v2/usercollection"
        self.headers = {
            "Authorization": f"Bearer {personal_access_token}"
        }
    
    def _make_request(self, endpoint, params=None):
        """Make API request with error handling"""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {endpoint}: {e}")
            return None
    
    def get_date_range(self, days_back=300):
        """Generate date range for API requests"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        return start_date.isoformat(), end_date.isoformat()
    
    def fetch_daily_sleep(self, start_date, end_date):
        """Fetch daily sleep data including HRV, temperature, respiratory rate"""
        params = {"start_date": start_date, "end_date": end_date}
        data = self._make_request("daily_sleep", params)
        if data and 'data' in data:
            return pd.DataFrame(data['data'])
        return pd.DataFrame()
    
    def fetch_daily_readiness(self, start_date, end_date):
        """Fetch daily readiness scores and contributing factors"""
        params = {"start_date": start_date, "end_date": end_date}
        data = self._make_request("daily_readiness", params)
        if data and 'data' in data:
            return pd.DataFrame(data['data'])
        return pd.DataFrame()
    
    def fetch_daily_activity(self, start_date, end_date):
        """Fetch daily activity and movement data"""
        params = {"start_date": start_date, "end_date": end_date}
        data = self._make_request("daily_activity", params)
        if data and 'data' in data:
            return pd.DataFrame(data['data'])
        return pd.DataFrame()
    
    def fetch_heart_rate(self, start_date, end_date):
        """Fetch detailed heart rate time series data"""
        params = {"start_date": start_date, "end_date": end_date}
        data = self._make_request("heartrate", params)
        if data and 'data' in data:
            return pd.DataFrame(data['data'])
        return pd.DataFrame()
    
    def fetch_daily_stress(self, start_date, end_date):
        """Fetch daily stress measurements"""
        params = {"start_date": start_date, "end_date": end_date}
        data = self._make_request("daily_stress", params)
        if data and 'data' in data:
            return pd.DataFrame(data['data'])
        return pd.DataFrame()
    
    def fetch_sleep_time_series(self, start_date, end_date):
        """Fetch detailed sleep time series (5-min intervals)"""
        params = {"start_date": start_date, "end_date": end_date}
        data = self._make_request("sleep", params)
        if data and 'data' in data:
            return pd.DataFrame(data['data'])
        return pd.DataFrame()
    
    def collect_all_data(self, days_back=300, output_dir="oura_data"):
        """
        Collect all available data types and save to CSV files
        
        Args:
            days_back: Number of days to fetch (default 300 for ~10 months)
            output_dir: Directory to save CSV files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get date range
        start_date, end_date = self.get_date_range(days_back)
        print(f"Fetching data from {start_date} to {end_date}")
        
        # Fetch all data types
        datasets = {
            "daily_sleep": self.fetch_daily_sleep,
            "daily_readiness": self.fetch_daily_readiness,
            "daily_activity": self.fetch_daily_activity,
            "heart_rate": self.fetch_heart_rate,
            "daily_stress": self.fetch_daily_stress,
            "sleep_time_series": self.fetch_sleep_time_series
        }
        
        results = {}
        for name, fetch_func in datasets.items():
            print(f"Fetching {name}...")
            df = fetch_func(start_date, end_date)
            if not df.empty:
                # Save to CSV
                filepath = os.path.join(output_dir, f"{name}.csv")
                df.to_csv(filepath, index=False)
                print(f"  ✓ Saved {len(df)} records to {filepath}")
                results[name] = df
            else:
                print(f"  ✗ No data available for {name}")
        
        return results
    
    def create_feature_summary(self, data_dict, output_dir="oura_data"):
        """Create a merged daily summary with key features for ML"""
        if 'daily_sleep' not in data_dict or 'daily_readiness' not in data_dict:
            print("Missing required datasets for feature summary")
            return None
        
        # Start with readiness (has most comprehensive daily summary)
        df = data_dict['daily_readiness'].copy()
        
        # Merge sleep data
        if not data_dict['daily_sleep'].empty:
            sleep_cols = ['day', 'score', 'contributors']
            sleep_df = data_dict['daily_sleep'][sleep_cols].copy()
            sleep_df = sleep_df.rename(columns={'score': 'sleep_score'})
            df = df.merge(sleep_df, on='day', how='left', suffixes=('', '_sleep'))
        
        # Merge activity data
        if 'daily_activity' in data_dict and not data_dict['daily_activity'].empty:
            activity_cols = ['day', 'score', 'contributors']
            activity_df = data_dict['daily_activity'][activity_cols].copy()
            activity_df = activity_df.rename(columns={'score': 'activity_score'})
            df = df.merge(activity_df, on='day', how='left', suffixes=('', '_activity'))
        
        # Merge stress data
        if 'daily_stress' in data_dict and not data_dict['daily_stress'].empty:
            stress_cols = ['day', 'stress_high', 'recovery_high']
            stress_df = data_dict['daily_stress'][stress_cols].copy()
            df = df.merge(stress_df, on='day', how='left')
        
        # Save merged summary
        filepath = os.path.join(output_dir, "daily_features.csv")
        df.to_csv(filepath, index=False)
        print(f"\n✓ Created feature summary with {len(df)} days: {filepath}")
        
        return df


def main():
    """Example usage"""
    # Get your Personal Access Token from: https://cloud.ouraring.com
    # Instructions: Log in → Personal Access Tokens → Create New Token
    
    token = input("Enter your Oura Personal Access Token: ").strip()
    
    if not token:
        print("Error: No token provided")
        return
    
    # Initialize collector
    collector = OuraDataCollector(token)
    
    # Collect all data (adjust days_back for your 9-10 months)
    print("\nCollecting Oura Ring data...")
    data = collector.collect_all_data(days_back=300)  # ~10 months
    
    # Create feature summary for ML
    if data:
        collector.create_feature_summary(data)
        print("\n✓ Data collection complete!")
        print("\nNext steps:")
        print("1. Create a symptom tracking spreadsheet with daily ratings")
        print("2. Merge your symptom data with oura_data/daily_features.csv")
        print("3. Start building your ML model")


if __name__ == "__main__":
    main()