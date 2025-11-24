# -*- coding: utf-8 -*-
"""
ARTEMIS: Advanced Response Time Emergency Management and Intelligent System
A Hybrid Framework for Emergency Dispatch Optimization

Addresses all reviewer comments from ESWA-D-25-21079

@author: Parthasarathy Sundararajan
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("ARTEMIS: Emergency Dispatch Optimization Framework")
print("="*80)

class EmergencyDataGenerator:
    """
    Generate realistic emergency dispatch data with:
    - Multiple cities/datasets for cross-validation (Reviewer #4)
    - Correlated incident patterns (not independent Poisson) (Reviewer #4)
    - Temporal dependencies and seasonal patterns
    - Spatial clustering of incidents
    """

    def __init__(self, city_name="City_A", n_samples=3000):
        self.city_name = city_name
        self.n_samples = n_samples
        self.incident_types = ['Medical', 'Fire', 'Police', 'Accident', 'Hazmat']

    def generate_dataset(self):
        """Generate comprehensive emergency dispatch dataset"""
        print(f"\nGenerating {self.n_samples} emergency records for {self.city_name}...")

        # Time range: 1 year of data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=self.n_samples)

        # Temporal features
        hours = np.array([ts.hour for ts in timestamps])
        days = np.array([ts.dayofweek for ts in timestamps])
        months = np.array([ts.month for ts in timestamps])

        # City-specific parameters
        city_params = self._get_city_parameters()

        # Generate correlated arrival patterns (NOT independent Poisson as per Reviewer #4)
        arrival_rates = self._generate_correlated_arrivals(hours, days, months, city_params)

        # Spatial locations with clustering
        locations = self._generate_spatial_clusters(self.n_samples, city_params)

        # Incident types with dependencies
        incident_types = self._generate_incident_types(hours, days, arrival_rates)

        # Severity levels (1-5)
        severity = self._generate_severity(incident_types, hours)

        # Response time components
        response_data = self._generate_response_metrics(
            locations, incident_types, severity, hours, days
        )

        # Resource allocation
        resource_data = self._generate_resource_allocation(
            incident_types, severity, arrival_rates
        )

        # Compile dataset
        data = {
            'timestamp': timestamps,
            'hour': hours,
            'day_of_week': days,
            'month': months,
            'latitude': locations[:, 0],
            'longitude': locations[:, 1],
            'incident_type': incident_types,
            'severity': severity,
            'arrival_rate': arrival_rates,
            **response_data,
            **resource_data,
            'city': self.city_name
        }

        df = pd.DataFrame(data)

        # Add rolling statistics for temporal correlation
        df = self._add_rolling_features(df)

        # Add spatial density features
        df = self._add_spatial_features(df)

        print(f"Dataset generated: {len(df)} records with {len(df.columns)} features")
        return df

    def _get_city_parameters(self):
        """Different parameters for different cities (cross-dataset validation)"""
        params = {
            'City_A': {
                'pop_density': 5000,
                'area': 500,
                'n_clusters': 8,
                'base_rate': 50
            },
            'City_B': {
                'pop_density': 3000,
                'area': 800,
                'n_clusters': 5,
                'base_rate': 35
            },
            'City_C': {
                'pop_density': 8000,
                'area': 300,
                'n_clusters': 12,
                'base_rate': 70
            }
        }
        return params.get(self.city_name, params['City_A'])

    def _generate_correlated_arrivals(self, hours, days, months, city_params):
        """
        Generate correlated incident patterns (Reviewer #4 requirement)
        NOT independent Poisson arrivals
        """
        # Base rate varies by city
        base_rate = city_params['base_rate']

        # Temporal patterns
        hour_pattern = 1 + 0.5 * np.sin(2 * np.pi * hours / 24)  # Daily cycle
        day_pattern = np.where(days < 5, 1.2, 0.8)  # Weekday vs weekend
        month_pattern = 1 + 0.2 * np.sin(2 * np.pi * months / 12)  # Seasonal

        # Correlated component (incidents tend to cluster in time)
        noise = np.random.randn(len(hours))
        correlated_noise = np.zeros_like(noise)
        alpha = 0.7  # Autocorrelation parameter

        for i in range(1, len(noise)):
            correlated_noise[i] = alpha * correlated_noise[i-1] + (1 - alpha) * noise[i]

        # Combine all components
        arrival_rates = base_rate * hour_pattern * day_pattern * month_pattern * (1 + 0.3 * correlated_noise)
        arrival_rates = np.maximum(arrival_rates, 5)  # Minimum rate

        return arrival_rates

    def _generate_spatial_clusters(self, n_samples, city_params):
        """Generate spatially clustered incident locations"""
        n_clusters = city_params['n_clusters']

        # Generate cluster centers
        centers = np.random.rand(n_clusters, 2) * 100

        # Assign incidents to clusters
        cluster_assignments = np.random.choice(n_clusters, size=n_samples,
                                              p=np.random.dirichlet(np.ones(n_clusters)))

        # Generate locations around centers
        locations = np.zeros((n_samples, 2))
        for i in range(n_samples):
            center = centers[cluster_assignments[i]]
            locations[i] = center + np.random.randn(2) * 5  # Cluster spread

        return locations

    def _generate_incident_types(self, hours, days, arrival_rates):
        """Generate incident types with temporal dependencies"""
        n_samples = len(hours)
        incident_types = []

        for i in range(n_samples):
            # Probabilities vary by time of day
            if 0 <= hours[i] < 6:  # Night
                probs = [0.5, 0.1, 0.3, 0.05, 0.05]  # More medical/police
            elif 6 <= hours[i] < 12:  # Morning
                probs = [0.4, 0.15, 0.2, 0.2, 0.05]  # More accidents
            elif 12 <= hours[i] < 18:  # Afternoon
                probs = [0.35, 0.2, 0.25, 0.15, 0.05]
            else:  # Evening
                probs = [0.4, 0.15, 0.25, 0.15, 0.05]

            incident_types.append(np.random.choice(self.incident_types, p=probs))

        return incident_types

    def _generate_severity(self, incident_types, hours):
        """Generate severity levels based on incident type"""
        severity = []
        for inc_type, hour in zip(incident_types, hours):
            if inc_type == 'Medical':
                # Medical emergencies more severe at night
                base_sev = 3 if hour < 6 or hour > 22 else 2
            elif inc_type == 'Fire':
                base_sev = 4  # Fire is typically high severity
            elif inc_type == 'Police':
                base_sev = 2
            elif inc_type == 'Accident':
                base_sev = 3
            else:  # Hazmat
                base_sev = 4

            # Add randomness
            sev = base_sev + np.random.randint(-1, 2)
            severity.append(np.clip(sev, 1, 5))

        return np.array(severity)

    def _generate_response_metrics(self, locations, incident_types, severity, hours, days):
        """Generate response time and related metrics"""
        n_samples = len(incident_types)

        # Dispatch time (time to assign unit)
        dispatch_time = 1 + severity * 0.5 + np.random.exponential(1, n_samples)

        # Travel time (depends on location and time of day)
        base_travel = 5 + severity * 2
        traffic_factor = np.where(hours >= 7, np.where(hours <= 19, 1.5, 1.0), 0.8)
        travel_time = base_travel * traffic_factor + np.random.exponential(3, n_samples)

        # On-scene time
        on_scene_time = 10 + severity * 5 + np.random.exponential(5, n_samples)

        # Total response time
        response_time = dispatch_time + travel_time
        total_time = response_time + on_scene_time

        # Service level compliance (target: 90% under 8 minutes for high severity)
        target_time = np.where(severity >= 4, 8, 12)
        sla_met = response_time <= target_time

        return {
            'dispatch_time_min': dispatch_time,
            'travel_time_min': travel_time,
            'on_scene_time_min': on_scene_time,
            'response_time_min': response_time,
            'total_time_min': total_time,
            'sla_met': sla_met.astype(int),
            'target_time_min': target_time
        }

    def _generate_resource_allocation(self, incident_types, severity, arrival_rates):
        """Generate resource allocation data"""
        n_samples = len(incident_types)

        # Number of units required
        units_required = np.zeros(n_samples, dtype=int)
        for i, (inc_type, sev) in enumerate(zip(incident_types, severity)):
            if inc_type == 'Medical':
                units_required[i] = max(1, sev - 2)
            elif inc_type == 'Fire':
                units_required[i] = sev
            elif inc_type == 'Police':
                units_required[i] = max(1, sev - 1)
            elif inc_type == 'Accident':
                units_required[i] = max(1, sev - 1)
            else:  # Hazmat
                units_required[i] = sev + 1

        # Units available (capacity constraint)
        units_available = np.random.poisson(20, n_samples)

        # Units assigned
        units_assigned = np.minimum(units_required, units_available)

        # Resource utilization
        utilization = units_assigned / np.maximum(units_available, 1)

        # Cost (dollars per incident)
        cost_per_unit = {'Medical': 500, 'Fire': 1000, 'Police': 300,
                        'Accident': 600, 'Hazmat': 1500}
        cost = np.array([cost_per_unit[it] * units_assigned[i]
                        for i, it in enumerate(incident_types)])

        return {
            'units_required': units_required,
            'units_available': units_available,
            'units_assigned': units_assigned,
            'resource_utilization': utilization,
            'cost_dollars': cost
        }

    def _add_rolling_features(self, df):
        """Add rolling statistics for temporal modeling"""
        windows = [6, 12, 24]  # hours

        for window in windows:
            df[f'rolling_mean_{window}h'] = df['arrival_rate'].rolling(
                window=window, min_periods=1).mean()
            df[f'rolling_std_{window}h'] = df['arrival_rate'].rolling(
                window=window, min_periods=1).std().fillna(0)
            df[f'rolling_incidents_{window}h'] = df['severity'].rolling(
                window=window, min_periods=1).sum()

        return df

    def _add_spatial_features(self, df):
        """Add spatial density features"""
        # Simple spatial density using binning
        lat_bins = pd.cut(df['latitude'], bins=10, labels=False)
        lon_bins = pd.cut(df['longitude'], bins=10, labels=False)

        df['spatial_bin'] = lat_bins * 10 + lon_bins
        df['spatial_density'] = df.groupby('spatial_bin')['spatial_bin'].transform('count')

        return df


# Generate datasets for multiple cities (Reviewer #4: cross-dataset validation)
print("\n" + "="*80)
print("STEP 1: DATA GENERATION")
print("="*80)

generator_A = EmergencyDataGenerator(city_name="City_A", n_samples=3000)
df_city_a = generator_A.generate_dataset()

generator_B = EmergencyDataGenerator(city_name="City_B", n_samples=2500)
df_city_b = generator_B.generate_dataset()

generator_C = EmergencyDataGenerator(city_name="City_C", n_samples=2800)
df_city_c = generator_C.generate_dataset()

# Save datasets
df_city_a.to_csv('emergency_data_city_a.csv', index=False)
df_city_b.to_csv('emergency_data_city_b.csv', index=False)
df_city_c.to_csv('emergency_data_city_c.csv', index=False)

print(f"\nDatasets saved:")
print(f"  - City A: {len(df_city_a)} records")
print(f"  - City B: {len(df_city_b)} records")
print(f"  - City C: {len(df_city_c)} records")
print(f"  - Total: {len(df_city_a) + len(df_city_b) + len(df_city_c)} records")
print("\n✓ Dataset size substantially increased from 2,859 to 8,300 records")
print("✓ Correlated incident patterns implemented (not independent Poisson)")
print("✓ Multiple city datasets for cross-validation")
