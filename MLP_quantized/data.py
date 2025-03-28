import pandas as pd
import os
import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import pickle

current_dir = os.path.dirname(__file__)
# Base directory containing the action folders
base_dir = '../actions'
# Output directory for the combined CSV file
output_dir = './data'

# List of action names (folder names)
actions = ['badminton', 'boxing', 'end', 'fencing', 'golf', 'reload', 'shield', 'snowbomb', 'walking', 'wearing']

def append_csv(output_csv):
    # List to store individual DataFrames
    df_list = []

    # Loop over each action folder
    for action in actions:
        # Create the full path to the action folder
        folder_path = os.path.join(current_dir, base_dir, action)
        # Glob for all CSV files in the current folder
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        
        for file in csv_files:
            # Read the CSV file with no header
            df = pd.read_csv(file, header=None)
            # Remove the Timestamp column
            df = df.iloc[:, 1:]
            # Add column names
            num_features = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']
            column_names = [f'{i}' for i in num_features]
            df.columns = column_names
            # Add a new column for the action label
            df['label'] = action
            # Append to our list of DataFrames
            df_list.append(df)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(df_list, ignore_index=True)
    # Save to CSV in output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    combined_df.to_csv(os.path.join(current_dir, output_dir, f"{output_csv}.csv"), index=False)
    print(f"Combined {len(combined_df)} actions into {output_csv}.csv.")

def extract_features_from_csv(file_path, window_size=40):
    # Load CSV
    df = pd.read_csv(file_path)
    
    # List of sensor columns (excluding label)
    sensor_cols = [col for col in df.columns if col != 'label']
    
    processed_data = []
    
    for i in range(0, len(df), window_size):
        chunk = df.iloc[i:i+window_size]
        
        # Check that all rows in the window have the same label, else print warning
        if not chunk['label'].nunique() == 1:
            print(f"Warning: Found a window with multiple labels: {chunk['label'].unique()}")
        if len(chunk) < window_size:
            continue
        
        features = {}
        
        for col in sensor_cols:
            data = chunk[col]
            # Time domain features
            features[f'{col}_mean'] = data.mean()
            features[f'{col}_std'] = data.std()  # Standard deviation for the sensor reading
            features[f'{col}_min'] = data.min()
            features[f'{col}_max'] = data.max()
            features[f'{col}_range'] = data.max() - data.min()
            features[f'{col}_median'] = data.median()
            # Use nan_to_num to prevent NaNs for skew and kurtosis
            features[f'{col}_skew'] = np.nan_to_num(skew(data), nan=0.0)
            features[f'{col}_kurtosis'] = np.nan_to_num(kurtosis(data), nan=0.0)
            features[f'{col}_mad'] = np.median(np.abs(data - np.median(data)))
            
            # Derivative-based features (approximate velocity)
            diff = np.diff(data)
            features[f'{col}_diff_mean'] = diff.mean()
            features[f'{col}_diff_std'] = diff.std()  # Standard deviation of the derivative
            
            # Frequency domain: FFT energy (excluding the DC component)
            fft_coeffs = fft(data)
            fft_energy = np.sum(np.abs(fft_coeffs[1:])**2)
            features[f'{col}_fft_energy'] = fft_energy
        
        
        features['label'] = chunk['label'].iloc[0]
        processed_data.append(features)

    # Convert to DataFrame
    result_df = pd.DataFrame(processed_data)
    # Save the transformed data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_df.to_csv(os.path.join(current_dir, output_dir, 'processed_data.csv'), index=False)
    print(f"Extracted {len(result_df)} actions from {file_path} and saved as processed_data.csv.")

def scale_data():
    # Load the processed data
    processed_csv = os.path.join(current_dir, output_dir, 'processed_data.csv')
    df = pd.read_csv(processed_csv)
    
    # Identify the feature columns (all columns except 'label')
    feature_cols = [col for col in df.columns if col != 'label']
    
    # Scale features to [0, 1] using MinMaxScaler
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save the scaled data
    scaled_csv = os.path.join(current_dir, output_dir, 'processed_data_scaled.csv')
    df.to_csv(scaled_csv, index=False)
    print(f"Saved scaled data with {len(df)} rows to {scaled_csv}.")
    
    # Export scaler using pickle for later use in deployment
    with open(os.path.join(current_dir, output_dir, 'minmax_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler exported to minmax_scaler.pkl.")

def main():
    # Combine all CSV files into one
    append_csv('combined_data')

    # Extract features from the combined data
    extract_features_from_csv(os.path.join(current_dir, output_dir, 'combined_data.csv'))

    # Scale the processed data
    scale_data()

if __name__ == '__main__':
    main()