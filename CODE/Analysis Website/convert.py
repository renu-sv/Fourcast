import pandas as pd
import os

# Read the stations data containing longitude and latitude
stations_df = pd.read_csv('stations.csv')

# Keep only the necessary columns from stations data
stations_location = stations_df[['Station Code', 'Latitude', 'Longitude']]

# Process each rivers file from 2012 to 2023
for year in range(2012, 2024):  # This will include 2012 through 2023
    file_path = f'data/seas/seas_{year}.csv'
    
    try:
        # Read the rivers data for this year
        seas_df = pd.read_csv(file_path)
        
        # Merge with stations data to add latitude and longitude
        # Assuming 'STATION CODE' in rivers data corresponds to 'Station Code' in stations data
        merged_df = pd.merge(
            seas_df,
            stations_location,
            left_on='STATION CODE',
            right_on='Station Code',
            how='left'
        )
        
        # If 'Station Code' column was added as a duplicate, remove it
        if 'Station Code' in merged_df.columns and 'STATION CODE' in merged_df.columns:
            merged_df = merged_df.drop(columns=['Station Code'])
        
        # Save the enriched data
        output_filename = f'seas_with_location_{year}.csv'
        merged_df.to_csv(output_filename, index=False)
        
        print(f"Processed {file_path} and saved to {output_filename}")
        
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Skipping this year.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("All files processed.")