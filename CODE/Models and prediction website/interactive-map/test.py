# Add these functions to handle existing CSV data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Function to load and combine data from multiple CSV files
def load_data_from_csv_files(data_directory):
    """
    Load data from multiple CSV files (one per year) and combine them into a single dataframe.
    Extract year from filename and add it as a column.
    Exclude specific files like 'a.csv' and 'b.csv'.
    """
    all_data = []
    
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
    
    # Exclude specific files
    exclude_files = ['a.csv', 'b.csv']
    csv_files = [f for f in csv_files if f not in exclude_files]
    
    print(f"Found {len(csv_files)} CSV files in {data_directory} after excluding {exclude_files}")
    
    for file_name in csv_files:
        file_path = os.path.join(data_directory, file_name)
        
        # Extract year from filename (assuming format contains '_2012.csv', '_2013.csv', etc.)
        try:
            year = int(file_name.rsplit('_', 1)[1].split('.')[0])
            print(f"Processing file: {file_name}, extracted year: {year}")
        except (IndexError, ValueError):
            print(f"Warning: Could not extract year from filename {file_name}, skipping this file")
            continue
        
        try:
            # Try with utf-8 encoding first
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            # If that fails, try with a different encoding
            try:
                df = pd.read_csv(file_path, encoding='latin1')
            except:
                # If both fail, try with the default encoding and error handling
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                print(f"Warning: Some lines in {file_name} were skipped due to encoding issues")
        
        # Add year column
        df['YEAR'] = year
        
        # Append to list of dataframes
        all_data.append(df)
    
    # Combine all dataframes
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined data shape: {combined_df.shape}")
        return combined_df
    else:
        print("No data files found!")
        return None

# Function to clean problematic characters and handle missing values
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

def train_and_validate_models(dataframe, station_code, validation_year=2023, years_to_predict=3):
    """
    Train models on data up to validation_year-1, validate on validation_year,
    then make future predictions for wqi column.
    """
    
    # The target column is always wqi
    target_column = 'wqi'
    
    # Filter data for the specific station
    station_data = dataframe[dataframe['STATION CODE'] == station_code].copy()
    
    # Sort by year
    station_data = station_data.sort_values('YEAR')
    
    # Split into training and validation sets
    train_data = station_data[station_data['YEAR'] < validation_year]
    validation_data = station_data[station_data['YEAR'] == validation_year]
    
    # Check if we have both training and validation data
    if train_data.empty:
        print(f"No training data available for station {station_code}")
        return None, None, None, None, None, None
    
    # Extract training features and target
    X_train = train_data[['YEAR']].copy()
    y_train = train_data[target_column].values
    
    # Create lag features for training data
    if len(y_train) > 1:
        # Create a lag_1 feature (previous year's value)
        X_train['lag_1'] = np.nan
        for i in range(1, len(y_train)):
            X_train.iloc[i, X_train.columns.get_loc('lag_1')] = y_train[i-1]
    
    # Drop rows with NaN (first row due to lag feature)
    X_train = X_train.dropna()
    y_train = y_train[1:] if len(y_train) > 1 else y_train
    
    # Check if we have enough data for training
    if len(X_train) < 3:
        print(f"Not enough data for station {station_code} to create predictions")
        return None, None, None, None, None, None
    
    # Prepare validation data if available
    validation_available = not validation_data.empty
    if validation_available:
        # For validation, we need the last value from the training set as lag_1
        X_val = validation_data[['YEAR']].copy()
        y_val = validation_data[target_column].values
        X_val['lag_1'] = y_train[-1]  # Last value from training set
    
    # 1. Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train[['YEAR', 'lag_1']], y_train)
    
    # 2. Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train[['YEAR', 'lag_1']], y_train)
    
    # 3. ARIMA Model
    try:
        # Extract just the target values for ARIMA
        arima_train = y_train
        
        # Fit ARIMA model (simple parameters for demonstration)
        arima_model = ARIMA(arima_train, order=(1, 1, 1))
        arima_model_fit = arima_model.fit()
    except Exception as e:
        print(f"Error fitting ARIMA model for station {station_code}: {e}")
        arima_model_fit = None
    
    # Calculate validation metrics if validation data available
    validation_metrics = None
    if validation_available:
        validation_metrics = {}
        
        # Linear Regression validation
        lr_val_pred = lr_model.predict(X_val[['YEAR', 'lag_1']])[0]
        validation_metrics['LR_Actual'] = y_val[0]
        validation_metrics['LR_Predicted'] = lr_val_pred
        validation_metrics['LR_Error'] = abs(y_val[0] - lr_val_pred)
        validation_metrics['LR_Percent_Error'] = abs((y_val[0] - lr_val_pred) / y_val[0] * 100) if y_val[0] != 0 else np.nan
        
        # Random Forest validation
        rf_val_pred = rf_model.predict(X_val[['YEAR', 'lag_1']])[0]
        validation_metrics['RF_Actual'] = y_val[0]
        validation_metrics['RF_Predicted'] = rf_val_pred
        validation_metrics['RF_Error'] = abs(y_val[0] - rf_val_pred)
        validation_metrics['RF_Percent_Error'] = abs((y_val[0] - rf_val_pred) / y_val[0] * 100) if y_val[0] != 0 else np.nan
        
        # ARIMA validation
        if arima_model_fit is not None:
            try:
                arima_val_pred = arima_model_fit.forecast(steps=1)[0]
                validation_metrics['ARIMA_Actual'] = y_val[0]
                validation_metrics['ARIMA_Predicted'] = arima_val_pred
                validation_metrics['ARIMA_Error'] = abs(y_val[0] - arima_val_pred)
                validation_metrics['ARIMA_Percent_Error'] = abs((y_val[0] - arima_val_pred) / y_val[0] * 100) if y_val[0] != 0 else np.nan
            except Exception as e:
                print(f"Error making ARIMA validation prediction: {e}")
    
    # Make future predictions
    last_year = station_data['YEAR'].max()
    last_value = station_data[station_data['YEAR'] == last_year][target_column].values[0]
    
    # Create dataframe for future predictions
    future_years = pd.DataFrame({'YEAR': range(last_year + 1, last_year + years_to_predict + 1)})
    future_years['lag_1'] = [last_value] + [0] * (years_to_predict - 1)  # Initialize with last known value
    
    # Make predictions with both models
    lr_predictions = []
    rf_predictions = []
    arima_predictions = []
    
    for i in range(years_to_predict):
        if i > 0:
            # Update lag value with the previous prediction
            future_years.loc[i, 'lag_1'] = lr_predictions[-1]
        
        # Make predictions for this year
        lr_pred = lr_model.predict(future_years.iloc[i:i+1][['YEAR', 'lag_1']])[0]
        rf_pred = rf_model.predict(future_years.iloc[i:i+1][['YEAR', 'lag_1']])[0]
        
        lr_predictions.append(lr_pred)
        rf_predictions.append(rf_pred)
    
    # ARIMA future predictions
    if arima_model_fit is not None:
        try:
            arima_forecast = arima_model_fit.forecast(steps=years_to_predict)
            arima_predictions = arima_forecast.tolist()
        except Exception as e:
            print(f"Error making ARIMA future predictions: {e}")
            arima_predictions = [np.nan] * years_to_predict
    else:
        arima_predictions = [np.nan] * years_to_predict
    
    # Store the predictions for the future years
    future_predictions = pd.DataFrame({
        'YEAR': range(last_year + 1, last_year + years_to_predict + 1),
        'LR_Prediction': lr_predictions,
        'RF_Prediction': rf_predictions,
        'ARIMA_Prediction': arima_predictions
    })
    
    # Return historical data along with other results
    return lr_model, rf_model, arima_model_fit, validation_metrics, future_predictions, station_data

def make_predictions_for_all_stations(processed_data, validation_year=2023, years_to_predict=3):
    """Make wqi predictions for all stations and return both predictions and historical data"""
    
    # Create a single consolidated DataFrame for future predictions
    all_future_predictions = pd.DataFrame()
    
    # Dictionary to store historical data for all stations
    all_historical_data = {}
    
    # List to store validation metrics for all stations
    all_validation_metrics = []
    
    # Get unique station codes
    station_codes = processed_data['STATION CODE'].unique()
    
    for station_code in station_codes:
        print(f"\nProcessing station {station_code}:")
        
        # Train and validate models
        lr_model, rf_model, arima_model, validation_metrics, future_predictions, historical_data = train_and_validate_models(
            processed_data, station_code, validation_year, years_to_predict
        )
        
        # Store historical data
        if historical_data is not None:
            all_historical_data[station_code] = historical_data
        
        # Store validation metrics
        if validation_metrics is not None:
            validation_metrics['STATION_CODE'] = station_code
            all_validation_metrics.append(validation_metrics)
        
        # Add future predictions to consolidated DataFrame
        if future_predictions is not None:
            # Add station code to predictions
            future_predictions['STATION_CODE'] = station_code
            
            # Get station state
            if 'STATE' in processed_data.columns:
                station_state = processed_data[processed_data['STATION CODE'] == station_code]['STATE'].iloc[0]
                future_predictions['STATE'] = station_state
            
            # Add to consolidated DataFrame
            all_future_predictions = pd.concat([all_future_predictions, future_predictions])
    
    # Convert validation metrics list to DataFrame
    validation_metrics_df = pd.DataFrame(all_validation_metrics) if all_validation_metrics else None
    
    return all_future_predictions, all_historical_data, validation_metrics_df

def plot_random_stations(historical_data_dict, predictions_df, validation_metrics_df=None, num_stations=5):
    """
    Plot wqi for a random selection of stations, showing both historical data and predictions.
    
    Parameters:
    - historical_data_dict: Dictionary with station codes as keys and historical dataframes as values
    - predictions_df: DataFrame containing predictions for all stations
    - validation_metrics_df: DataFrame containing validation metrics for all stations
    - num_stations: Number of random stations to plot
    """
    # Get list of station codes that have both historical data and predictions
    valid_stations = []
    for station in historical_data_dict.keys():
        if station in predictions_df['STATION_CODE'].unique():
            valid_stations.append(station)
    
    # Check if we have enough valid stations
    if len(valid_stations) < num_stations:
        print(f"Warning: Only {len(valid_stations)} stations have both historical data and predictions")
        num_stations = len(valid_stations)
    
    # Randomly select stations
    if valid_stations:
        random_stations = random.sample(valid_stations, num_stations)
        
        # Create a figure with subplots
        fig, axes = plt.subplots(num_stations, 1, figsize=(12, num_stations * 4), sharex=False)
        
        # If only one station, axes needs to be in a list for consistency
        if num_stations == 1:
            axes = [axes]
        
        # Plot for each random station
        for i, station_code in enumerate(random_stations):
            ax = axes[i]
            
            # Get historical data for this station
            hist_data = historical_data_dict[station_code]
            
            # Get predictions for this station
            station_predictions = predictions_df[predictions_df['STATION_CODE'] == station_code]
            
            # Plot historical data
            ax.plot(hist_data['YEAR'], hist_data['wqi'], 'o-', color='blue', label='Historical wqi')
            
            # Get validation metrics for this station if available
            validation_points = {}
            if validation_metrics_df is not None:
                station_validation = validation_metrics_df[validation_metrics_df['STATION_CODE'] == station_code]
                if not station_validation.empty:
                    # Extract validation year predictions
                    validation_year = hist_data['YEAR'].max()  # Assuming validation year is the last year in historical data
                    
                    if 'LR_Predicted' in station_validation.columns:
                        validation_points['LR'] = station_validation['LR_Predicted'].iloc[0]
                        ax.plot(validation_year, validation_points['LR'], 's', color='green', markersize=10, 
                                label='LR 2023 Prediction')
                    
                    if 'RF_Predicted' in station_validation.columns:
                        validation_points['RF'] = station_validation['RF_Predicted'].iloc[0]
                        ax.plot(validation_year, validation_points['RF'], '^', color='red', markersize=10, 
                                label='RF 2023 Prediction')
                    
                    if 'ARIMA_Predicted' in station_validation.columns:
                        validation_points['ARIMA'] = station_validation['ARIMA_Predicted'].iloc[0]
                        ax.plot(validation_year, validation_points['ARIMA'], 'd', color='orange', markersize=10, 
                                label='ARIMA 2023 Prediction')
            
            # Plot future predictions
            future_years = station_predictions['YEAR']
            ax.plot(future_years, station_predictions['LR_Prediction'], 's--', 
                    color='green', label='Linear Regression')
            ax.plot(future_years, station_predictions['RF_Prediction'], '^--', 
                    color='red', label='Random Forest')
            ax.plot(future_years, station_predictions['ARIMA_Prediction'], 'd--', 
                    color='orange', label='ARIMA')
            
            # Add titles and labels
            station_name = f"Station {station_code}"
            if 'STATE' in hist_data.columns:
                state = hist_data['STATE'].iloc[0]
                station_name += f" ({state})"
            
            ax.set_title(f"wqi for {station_name}")
            ax.set_xlabel('Year')
            ax.set_ylabel('Water Quality Index (wqi)')
            ax.grid(True)
            ax.legend()
            
            # Set x-axis ticks to show all years
            all_years = sorted(list(hist_data['YEAR'].unique()) + list(future_years))
            ax.set_xticks(all_years)
            
            # Add a text annotation with prediction values
            text_str = ""
            
            # Add validation metrics if available
            if validation_points:
                validation_year = hist_data['YEAR'].max()
                actual_value = hist_data[hist_data['YEAR'] == validation_year]['wqi'].iloc[0]
                text_str += f"2023 Validation:\nActual={actual_value:.2f}\n"
                
                if 'LR' in validation_points:
                    lr_error = abs(actual_value - validation_points['LR'])
                    lr_pct_error = abs((actual_value - validation_points['LR']) / actual_value * 100) if actual_value != 0 else float('nan')
                    text_str += f"LR={validation_points['LR']:.2f} (Error: {lr_error:.2f}, {lr_pct_error:.2f}%)\n"
                
                if 'RF' in validation_points:
                    rf_error = abs(actual_value - validation_points['RF'])
                    rf_pct_error = abs((actual_value - validation_points['RF']) / actual_value * 100) if actual_value != 0 else float('nan')
                    text_str += f"RF={validation_points['RF']:.2f} (Error: {rf_error:.2f}, {rf_pct_error:.2f}%)\n"
                
                if 'ARIMA' in validation_points:
                    arima_error = abs(actual_value - validation_points['ARIMA'])
                    arima_pct_error = abs((actual_value - validation_points['ARIMA']) / actual_value * 100) if actual_value != 0 else float('nan')
                    text_str += f"ARIMA={validation_points['ARIMA']:.2f} (Error: {arima_error:.2f}, {arima_pct_error:.2f}%)\n"
                
                text_str += "\n"
            
            # Add future predictions
            text_str += f"Future Predictions:\n"
            for year, lr, rf, arima in zip(
                station_predictions['YEAR'],
                station_predictions['LR_Prediction'],
                station_predictions['RF_Prediction'],
                station_predictions['ARIMA_Prediction']
            ):
                text_str += f"Year {year}: LR={lr:.2f}, RF={rf:.2f}, ARIMA={arima:.2f}\n"
            
            # Position the text box in the upper right corner
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig('random_stations_wqi_predictions.png')
        plt.show()
        print(f"Plot saved as 'random_stations_wqi_predictions.png'")
    else:
        print("No valid stations to plot")
def load_existing_csv(filename):
    """Load existing CSV file if it exists, return empty DataFrame if not"""
    try:
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            print(f"Loaded existing data from {filename}: {len(existing_df)} records")
            return existing_df
        else:
            print(f"No existing file found at {filename}, will create new file")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading existing file {filename}: {e}")
        return pd.DataFrame()

def append_to_csv(df, filename, key_columns):
    """
    Append new data to existing CSV file, avoiding duplicates and retaining original contents.
    
    Parameters:
    - df: DataFrame with new data to append
    - filename: Target CSV file name
    - key_columns: List of column names that identify unique records
    """
    if df.empty:
        print(f"No new data to append to {filename}")
        return
    
    # Load existing data
    existing_df = load_existing_csv(filename)
    
    if existing_df.empty:
        # No existing file, just save the new data
        df.to_csv(filename, index=False)
        print(f"Created new file {filename} with {len(df)} records")
        return
    
    # Check column compatibility
    missing_columns = [col for col in df.columns if col not in existing_df.columns]
    if missing_columns:
        print(f"Warning: New data contains columns not in existing file: {missing_columns}")
        # Add missing columns to existing dataframe
        for col in missing_columns:
            existing_df[col] = None
    
    # Ensure all columns in the new data are present in the existing data
    for col in existing_df.columns:
        if col not in df.columns:
            df[col] = None
    
    # Combine existing and new data
    combined_df = pd.concat([existing_df, df], ignore_index=True)
    
    # Remove duplicates based on key columns
    combined_df = combined_df.drop_duplicates(subset=key_columns)
    
    # Save the combined data back to the file
    combined_df.to_csv(filename, index=False)
    print(f"Updated {filename} with {len(df)} new records. Total records: {len(combined_df)}")

# Modify the main execution block to use these functions
if __name__ == "__main__":
    # Set the directory containing your CSV files
    data_directory = '/home/acer/Music/DPPRO/rivers_with_locations/rlwqi'  # Update this to your directory path
    
    # Load and combine data from all CSV files
    combined_df = load_data_from_csv_files(data_directory)
    
    if combined_df is not None:
        # Display initial data info
        print("\nInitial data info:")
        print(combined_df.info())
        
        # Preprocess the data specifically for wqi prediction
        processed_df = preprocess_data(combined_df)
        
        if processed_df is not None:
            # Display the processed data info
            print("\nProcessed data info:")
            print(processed_df.info())
            
            # Make wqi predictions for all stations
            validation_year = 2023  # Year to use for validation
            years_to_predict = 3    # How many years into the future
            
            all_future_predictions, all_historical_data, validation_metrics_df = make_predictions_for_all_stations(
                processed_df, validation_year=validation_year, years_to_predict=years_to_predict
            )
            
            # Save validation metrics to a CSV file, appending new data
            if validation_metrics_df is not None and not validation_metrics_df.empty:
                # Define key columns for validation metrics
                validation_key_columns = ['STATION_CODE']
                
                # Append to CSV
                append_to_csv(validation_metrics_df, '/home/acer/Music/DPPRO/rivers_with_locations/rlwqi/b.csv', validation_key_columns)
            
            # Save wqi predictions, appending new data
            if not all_future_predictions.empty:
                # Sort by station code and year for better readability
                all_future_predictions = all_future_predictions.sort_values(['STATION_CODE', 'YEAR'])
                
                # Define key columns for predictions
                prediction_key_columns = ['STATION_CODE', 'YEAR']
                
                # Append to CSV
                append_to_csv(all_future_predictions, '/home/acer/Music/DPPRO/rivers_with_locations/rlwqi/a.csv', prediction_key_columns)
                
                # Plot wqi for random stations
                print("\nGenerating plots for 5 random stations...")
                plot_random_stations(all_historical_data, all_future_predictions, validation_metrics_df, num_stations=5)
        else:
            print("Failed to preprocess the data. Please check if the wqi column exists.")
    else:
        print("No data to process. Please check your data directory.")