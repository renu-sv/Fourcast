from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import base64
from geopy.distance import geodesic
from math import radians, sin, cos, sqrt, atan2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

app = Flask(__name__)
stations_df = pd.read_excel("stations.xlsx")
stations_df.columns = stations_df.columns.str.strip()

# Load prediction data
predictions_df = pd.read_csv("/home/acer/Music/DPPRO/rivers_with_locations/rlwqi/wqi_predictions.csv")
validation_df = pd.read_csv("/home/acer/Music/DPPRO/rivers_with_locations/rlwqi/validation_metrics.csv")

print(predictions_df.head())  # Print the first few rows of predictions_df
print(predictions_df.columns)  # Print the column names

# Define the data directory
data_directory = '/home/acer/Music/DPPRO/rivers_with_locations/rlwqi'

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lat2 - lon1)
    a = sin(d_lat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c
def preprocess_data(dataframe):
    """Clean problematic characters and handle missing values specifically for wqi column"""
    
    # Make a copy to avoid warnings
    df_processed = dataframe.copy()
    
    # Check if wqi column exists
    if 'wqi' not in df_processed.columns:
        print("wqi column not found in the dataset. Available columns are:")
        print(df_processed.columns.tolist())
        return None
    
    # Replace problematic characters in the wqi column
    df_processed['wqi'] = df_processed['wqi'].astype(str)
    
    # Replace problematic characters with NaN
    problematic_chars = ['�', '#VALUE!', '#DIV/0!', 'None', 'nan', 'NA', 'N/A', '']
    for char in problematic_chars:
        df_processed['wqi'] = df_processed['wqi'].replace(char, np.nan)
    
    # Convert wqi to numeric, forcing errors to NaN
    df_processed['wqi'] = pd.to_numeric(df_processed['wqi'], errors='coerce')
    
    print(f"Number of NaN values in wqi column after initial processing: {df_processed['wqi'].isna().sum()}")
    
    # Group by station code to fill missing values with station-specific means
    for station in df_processed['STATION CODE'].unique():
        station_mask = df_processed['STATION CODE'] == station
        
        # Get mean for this station
        station_mean = df_processed.loc[station_mask, 'wqi'].mean()
        
        # If all values for this station are NaN, use overall mean
        if pd.isna(station_mean):
            station_mean = df_processed['wqi'].mean()
        
        # Fill missing values
        df_processed.loc[station_mask, 'wqi'] = df_processed.loc[station_mask, 'wqi'].fillna(station_mean)
    
    # Check for any remaining NaN values in wqi
    nan_count = df_processed['wqi'].isna().sum()
    if nan_count > 0:
        print(f"Warning: There are still {nan_count} NaN values in wqi column after preprocessing")
        # Fill any remaining NaNs with the overall mean
        df_processed['wqi'] = df_processed['wqi'].fillna(df_processed['wqi'].mean())
    
    return df_processed

def load_historical_data_for_station(station_code):
    """Load historical data for a specific station from CSV files."""
    all_data = []
    
    # Ensure station_code is an integer for comparison
    station_code = int(station_code)
    
    # Check if the data directory exists
    if not os.path.exists(data_directory):
        print(f"Data directory '{data_directory}' does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame if the directory doesn't exist
    
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
    
    for file_name in csv_files:
        # Skip prediction or validation files
        if 'prediction' in file_name.lower() or 'validation' in file_name.lower():
            continue
        
        try:
            # Extract year from the filename (assuming the year is at the end of the filename)
            year = int(file_name.rsplit('_', 1)[1].split('.')[0])
            print(f"Extracting year: {year} from file: {file_name}")
        except (ValueError, IndexError):
            print(f"Warning: Could not extract year from filename: {file_name}. Skipping file.")
            continue
        
        file_path = os.path.join(data_directory, file_name)
        
        try:
            # Try reading the CSV file with utf-8 encoding
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # Fallback to latin1 encoding if utf-8 fails
                df = pd.read_csv(file_path, encoding='latin1')
            except Exception as e:
                print(f"Error reading file {file_name}: {e}. Skipping file.")
                continue
        
        # Preprocess the data to clean and handle missing values
        # df = preprocess_data(df)
        
        # Check if the station code column exists
        station_col = next((col for col in df.columns if 'STATION CODE' in col.upper()), None)
        if not station_col:
            print(f"Warning: No 'STATION CODE' column found in {file_name}. Skipping file.")
            continue
        
        # Ensure the station code column is an integer for comparison
        df[station_col] = pd.to_numeric(df[station_col], errors='coerce').fillna(-1).astype(int)
        
        # Filter the data for the specific station
        station_data = df[df[station_col] == station_code].copy()  # Use .copy() to avoid SettingWithCopyWarning
        
        if not station_data.empty:
            print(f"Station data found for station code {station_code} in file {file_name}.")
            
            # Add the year column to the station data
            station_data.loc[:, 'YEAR'] = year  # Use .loc to avoid SettingWithCopyWarning
            
            # Check if the WQI column exists
            wqi_col = next((col for col in df.columns if col.lower() == 'wqi'), None)
            if wqi_col:
                # Keep only the relevant columns: station code, WQI, and year
                station_data = station_data[[station_col, wqi_col, 'YEAR']]
                all_data.append(station_data)
            else:
                print(f"Warning: No 'WQI' column found in {file_name}. Skipping WQI extraction.")
    
    # Combine all the dataframes into one
    if all_data:
        print('Combining data from multiple files.')
        combined_df = pd.concat(all_data, ignore_index=True)
        # Standardize column names for consistency
        combined_df.columns = ['STATION_CODE', 'wqi', 'YEAR']
        return combined_df
    else:
        print(f"No data found for station code {station_code}.")
        return pd.DataFrame()  # Return an empty DataFrame if no data is found

def generate_wqi_plot(station_code):
    """Generate a WQI plot for a specific station"""
    # Load historical data
    historical_data = load_historical_data_for_station(station_code)
    
    # Get station details
    station_info = stations_df[stations_df['Station Code'] == station_code]
    station_name = station_info['Station Name'].iloc[0] if not station_info.empty else f"Station {station_code}"
    
    # Check if we have any historical data
    if historical_data.empty:
        return None, "No historical data found for this station"
    
    # Ensure station_code is an integer
    station_code = int(station_code)
    
    # Get future predictions
    # Get future predictions
    print(f"Data type of STATION_CODE in predictions_df: {predictions_df['STATION_CODE'].dtype}")
    future_data = predictions_df[predictions_df['STATION_CODE'] == station_code]
    # if future_data.empty:
    #     return None, "No future predictions found for this station"
    # Get validation predictions for 2023
    validation_data = validation_df[validation_df['STATION_CODE'] == station_code]
    
    # Sort historical data by year
    historical_data = historical_data.sort_values(by='YEAR')
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot historical data
    plt.plot(historical_data['YEAR'], historical_data['wqi'], 'o-', color='blue', label='Historical WQI')
    
    # Plot validation predictions (2023)
    if not validation_data.empty:
        validation_year = historical_data['YEAR'].max()  # Assuming validation year is the last historical year
        
        if 'LR_Predicted' in validation_data.columns:
            plt.plot(validation_year, validation_data['LR_Predicted'].iloc[0], 's', color='green', markersize=10, 
                     label='LR 2023 Prediction')
        
        if 'RF_Predicted' in validation_data.columns:
            plt.plot(validation_year, validation_data['RF_Predicted'].iloc[0], '^', color='red', markersize=10, 
                     label='RF 2023 Prediction')
        
        if 'ARIMA_Predicted' in validation_data.columns:
            plt.plot(validation_year, validation_data['ARIMA_Predicted'].iloc[0], 'd', color='orange', markersize=10, 
                     label='ARIMA 2023 Prediction')
    
    # Plot future predictions
    if not future_data.empty:
        future_data = future_data.sort_values(by='YEAR')  # Sort future predictions by year
        plt.plot(future_data['YEAR'], future_data['LR_Prediction'], 's--', 
                 color='green', label='Linear Regression')
        plt.plot(future_data['YEAR'], future_data['RF_Prediction'], '^--', 
                 color='red', label='Random Forest')
        plt.plot(future_data['YEAR'], future_data['ARIMA_Prediction'], 'd--', 
                 color='orange', label='ARIMA')
    
    # Set labels and title
    plt.title(f"Water Quality Index for {station_name}")
    plt.xlabel('Year')
    plt.ylabel('Water Quality Index (WQI)')
    plt.grid(True)
    plt.legend()
    
    # Create text annotation with prediction details
    text_str = ""
    
    # Add validation metrics if available
    if not validation_data.empty:
        validation_year = historical_data['YEAR'].max()
        actual_value = historical_data[historical_data['YEAR'] == validation_year]['wqi'].iloc[0]
        text_str += f"2023 Validation:\nActual={actual_value:.2f}\n"
        
        if 'LR_Predicted' in validation_data.columns:
            lr_pred = validation_data['LR_Predicted'].iloc[0]
            lr_error = abs(actual_value - lr_pred)
            lr_pct_error = abs((actual_value - lr_pred) / actual_value * 100) if actual_value != 0 else float('nan')
            text_str += f"LR={lr_pred:.2f} (Error: {lr_error:.2f}, {lr_pct_error:.2f}%)\n"
        
        if 'RF_Predicted' in validation_data.columns:
            rf_pred = validation_data['RF_Predicted'].iloc[0]
            rf_error = abs(actual_value - rf_pred)
            rf_pct_error = abs((actual_value - rf_pred) / actual_value * 100) if actual_value != 0 else float('nan')
            text_str += f"RF={rf_pred:.2f} (Error: {rf_error:.2f}, {rf_pct_error:.2f}%)\n"
        
        if 'ARIMA_Predicted' in validation_data.columns:
            arima_pred = validation_data['ARIMA_Predicted'].iloc[0]
            arima_error = abs(actual_value - arima_pred)
            arima_pct_error = abs((actual_value - arima_pred) / actual_value * 100) if actual_value != 0 else float('nan')
            text_str += f"ARIMA={arima_pred:.2f} (Error: {arima_error:.2f}, {arima_pct_error:.2f}%)\n"
        
        text_str += "\n"
    
    # Add future predictions
    if not future_data.empty:
        text_str += f"Future Predictions:\n"
        for _, row in future_data.iterrows():
            text_str += f"Year {int(row['YEAR'])}: LR={row['LR_Prediction']:.2f}, RF={row['RF_Prediction']:.2f}, ARIMA={row['ARIMA_Prediction']:.2f}\n"
    
    # Add text annotation to plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=9, 
             verticalalignment='top', bbox=props)
    
    # Convert plot to base64 encoded image
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    
    return plot_url, None

def get_valid_station_codes(data_directory):
    """Get all unique station codes from the CSV files in the directory."""
    valid_station_codes = set()
    
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
    
    for file_name in csv_files:
        file_path = os.path.join(data_directory, file_name)
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Check if the 'STATION_CODE' column exists
            if 'STATION_CODE' in df.columns:
                valid_station_codes.update(df['STATION_CODE'].dropna().unique())
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
    
    return valid_station_codes

# Load station data
stations_df = pd.read_excel("stations.xlsx")
stations_df.columns = stations_df.columns.str.strip()

# Get valid station codes from the CSV files
valid_station_codes = get_valid_station_codes("/home/acer/Music/DPPRO/rivers_with_locations/rlwqi")

# Filter stations_df to include only stations with valid station codes
stations_df = stations_df[stations_df['Station Code'].isin(valid_station_codes)]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stations")
def stations():
    features = []
    for _, row in stations_df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["Longitude"], row["Latitude"]],
            },
            "properties": {
                "station": row["Station Name"]
            }
        })
    return jsonify({
        "type": "FeatureCollection",
        "features": features
    })
    
@app.route('/get_all_stations', methods=['GET'])
def get_all_stations():
    stations = []

    for _, row in stations_df.iterrows():
        stations.append({
            "station_code": row['Station Code'],
            "latitude": row['Latitude'],
            "longitude": row['Longitude'],
            "Wbodytype": row['Wbodytype']
        })

    return jsonify(stations)

@app.route('/get_nearest_station', methods=['POST'])
def get_nearest_station():
    data = request.json
    click_lat = data['lat']
    click_lon = data['lon']
    
    # Calculate distance to all stations
    stations_df['distance'] = stations_df.apply(
        lambda row: haversine(click_lat, click_lon, row['Latitude'], row['Longitude']), axis=1
    )
    
    # Find the nearest station
    nearest_station = stations_df.loc[stations_df['distance'].idxmin()]
    
    # Limit interpolation distance
    MAX_DISTANCE_KM = 20  # You can change this threshold

    if nearest_station['distance'] > MAX_DISTANCE_KM:
        return jsonify({'error': f"No nearby station within {MAX_DISTANCE_KM} km"}), 404

    # Generate WQI plot for this station
    station_code = str(nearest_station['Station Code'])
    plot_url, error_msg = generate_wqi_plot(station_code)
    
    response = {
        "station_code": station_code,
        "station_name": nearest_station['Station Name'],
        "latitude": float(nearest_station['Latitude']),
        "longitude": float(nearest_station['Longitude']),
        "distance_km": round(float(nearest_station['distance']), 2)
    }
    
    if plot_url:
        response["plot_data"] = plot_url
    elif error_msg:
        response["error_msg"] = error_msg
    
    return jsonify(response)

@app.route('/get_station_wqi/<station_code>', methods=['GET'])
def get_station_wqi(station_code):
    """Get WQI plot for a specific station"""
    plot_url, error_msg = generate_wqi_plot(station_code)
    
    if plot_url:
        return jsonify({"plot_data": plot_url})
    else:
        return jsonify({"error": error_msg or "Failed to generate plot"}), 404

if __name__ == "__main__":
    app.run(debug=True)