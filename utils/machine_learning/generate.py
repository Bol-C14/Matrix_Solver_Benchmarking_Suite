import numpy as np
import pandas as pd

# Parameters for the dataset
np.random.seed(42)  # Ensure reproducibility
num_samples_defective = 100
num_samples_non_defective = 100

# Generate synthetic data
# Defective items
weights_defective = np.random.normal(50, 5, num_samples_defective)
volumes_defective = np.random.normal(30, 3, num_samples_defective)
labels_defective = np.ones(num_samples_defective)

# Non-defective items
weights_non_defective = np.random.normal(70, 5, num_samples_non_defective)
volumes_non_defective = np.random.normal(50, 3, num_samples_non_defective)
labels_non_defective = np.zeros(num_samples_non_defective)

# Combine the data
weights = np.concatenate([weights_defective, weights_non_defective])
volumes = np.concatenate([volumes_defective, volumes_non_defective])
labels = np.concatenate([labels_defective, labels_non_defective])

# Create a DataFrame
data = pd.DataFrame({
    'Weight': weights,
    'Volume': volumes,
    'Label': labels
})

# Save to CSV file
output_file = 'synthetic_dataset.csv'
data.to_csv(output_file, index=False)

print(f"Dataset generated and saved to {output_file}")
