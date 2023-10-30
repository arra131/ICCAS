import numpy as np
import pandas as pd
import pickle

# Define the number of timelines (n) and the maximum timeline length
n = 5
max_timeline_length = 1000

# Create empty DataFrames to store the timelines
discrete_timeline_data = pd.DataFrame()
continuous_timeline_data = pd.DataFrame()

# Generate n timelines
for _ in range(n):
    # Generate a random timeline length (between 1 and max_timeline_length)
    timeline_length = np.random.randint(1, max_timeline_length + 1)

    # Generate discrete features (3 columns)
    discrete_features = {
        'DiscreteFeature1': np.random.choice(['A', 'B', 'C'], size=timeline_length),
        'DiscreteFeature2': np.random.choice(['X', 'Y', 'Z'], size=timeline_length),
        'DiscreteFeature3': np.random.choice([1, 2, 3], size=timeline_length)
    }

    # Introduce simple rules and randomness to relate the discrete features
    for i in range(1, timeline_length):
        if np.random.rand() < 0.2:  # 20% chance to apply a rule
            # Apply a rule: If 'A' in DiscreteFeature1, set 'X' in DiscreteFeature2
            if discrete_features['DiscreteFeature1'][i - 1] == 'A':
                discrete_features['DiscreteFeature2'][i] = 'X'
            # Apply a rule: If 'C' in DiscreteFeature1, set 3 in DiscreteFeature3
            if discrete_features['DiscreteFeature1'][i - 1] == 'C':
                discrete_features['DiscreteFeature3'][i] = 3

    # Generate continuous features (3 columns)
    continuous_features = {
        'ContinuousFeature1': np.random.uniform(0, 1, size=timeline_length),
        'ContinuousFeature2': np.random.uniform(0, 1, size=timeline_length),
        'ContinuousFeature3': np.random.uniform(0, 1, size=timeline_length)
    }

    # Create DataFrames for this timeline
    discrete_timeline = pd.DataFrame(discrete_features)
    continuous_timeline = pd.DataFrame(continuous_features)

    # Append the timelines to the main DataFrames
    discrete_timeline_data = pd.concat([discrete_timeline_data, discrete_timeline], ignore_index=True)
    continuous_timeline_data = pd.concat([continuous_timeline_data, continuous_timeline], ignore_index=True)

# Define the correlation matrix for continuous features
correlation_matrix = np.array([
    [1.0, 0.3, 0.2],
    [0.3, 1.0, -0.1],
    [0.2, -0.1, 1.0]
])

# Apply the correlations to the continuous features in the continuous_timeline_data
continuous_columns = ['ContinuousFeature1', 'ContinuousFeature2', 'ContinuousFeature3']
continuous_timeline_data[continuous_columns] = np.dot(continuous_timeline_data[continuous_columns], correlation_matrix)

# Save the DataFrames to separate pickle files
discrete_timeline_data.to_pickle('discrete_data.pkl')
continuous_timeline_data.to_pickle('continuous_data.pkl')

# Print the generated data
print(discrete_timeline_data)
print(continuous_timeline_data)