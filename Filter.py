import pandas as pd

# Assuming df is your DataFrame
file_path = "/Users/ISASC_ST/Documents/Data Project/americans_by_descent.csv"
df = pd.read_csv(file_path)
# List of columns to keep
columns_to_keep = ['African', 'European', 'Asian', 'name']  # Add other columns you want to keep

# Filter the DataFrame to keep only these columns
df_filtered = df[columns_to_keep]
df_filtered.to_csv('filtered_dataset.csv', index=False)
# Now df_filtered contains only the columns specified in columns_to_keep
