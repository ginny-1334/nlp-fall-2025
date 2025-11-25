import pandas as pd
import glob
import json

# Get all CSV files in the current directory
csv_files = glob.glob('*.csv')

# Initialize an empty list to hold DataFrames
dfs = []

# Read each CSV and append to the list
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(dfs, ignore_index=True)

# Remove duplicates based on 'Link' column
combined_df = combined_df.drop_duplicates(subset='Job Link')

print(f"Total job posts: {len(combined_df)}")

# Convert to JSON
combined_json = combined_df.to_json(orient='records')

# Save to a single JSON file
with open('combined_data.json', 'w') as f:
    f.write(combined_json)

print("All CSV files combined into combined_data.json with duplicates removed by Link")