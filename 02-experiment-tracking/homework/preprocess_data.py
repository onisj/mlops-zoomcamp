import os
import pickle
import logging
import click
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def dump_pickle(obj, filename: str):
    """Save object to pickle file."""
    try:
        with open(filename, "wb") as f_out:
            pickle.dump(obj, f_out)
        logging.info(f"Saved {filename}")
    except Exception as e:
        logging.error(f"Failed to save {filename}: {e}")
        raise

def read_dataframe(filename: str, dataset: str = "green"):
    """Read and preprocess Parquet file."""
    if not os.path.exists(filename):
        logging.error(f"File not found: {filename}")
        raise FileNotFoundError(f"File not found: {filename}")
    
    try:
        logging.info(f"Loading {filename}")
        df = pd.read_parquet(filename)
        
        # Select datetime columns based on dataset
        if dataset == "yellow":
            pickup_col = 'tpep_pickup_datetime'
            dropoff_col = 'tpep_dropoff_datetime'
        else:  # green
            pickup_col = 'lpep_pickup_datetime'
            dropoff_col = 'lpep_dropoff_datetime'
        
        # Compute duration in minutes
        df['duration'] = df[dropoff_col] - df[pickup_col]
        df['duration'] = df['duration'].apply(lambda td: td.total_seconds() / 60)
        
        # Filter durations
        original_len = len(df)
        df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
        logging.info(f"Filtered {original_len - len(df)} rows with duration outside 1–60 minutes")
        
        # Handle missing values
        df = df.dropna(subset=['PULocationID', 'DOLocationID', 'trip_distance'])
        logging.info(f"Removed {original_len - len(df)} rows with missing values")
        
        # Convert categorical features to strings
        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        
        return df
    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")
        raise

def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    """Preprocess DataFrame and apply DictVectorizer."""
    try:
        # Create combined feature
        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
        categorical = ['PU_DO']
        numerical = ['trip_distance']
        
        # Convert to dictionary for vectorization
        dicts = df[categorical + numerical].to_dict(orient='records')
        
        # Apply DictVectorizer
        if fit_dv:
            X = dv.fit_transform(dicts)
            logging.info("Fitted DictVectorizer")
        else:
            X = dv.transform(dicts)
            logging.info("Transformed data with DictVectorizer")
        
        return X, dv
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise

@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the raw NYC taxi trip data was saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
@click.option(
    "--dataset",
    default="green",
    help="Dataset type (e.g., green, yellow)"
)
def run_data_prep(raw_data_path: str, dest_path: str, dataset: str = "green"):
    """Preprocess NYC taxi trip data and save results."""
    try:
        # Validate dataset
        if dataset not in ["green", "yellow"]:
            logging.error(f"Invalid dataset: {dataset}. Use 'green' or 'yellow'")
            raise ValueError(f"Invalid dataset: {dataset}")
        
        # Load parquet files
        for month in ['01', '02', '03']:
            file_path = os.path.join(raw_data_path, f"{dataset}_tripdata_2023-{month}.parquet")
            if not os.path.exists(file_path):
                logging.error(f"Missing file: {file_path}")
                raise FileNotFoundError(f"Missing file: {file_path}")
        
        df_train = read_dataframe(os.path.join(raw_data_path, f"{dataset}_tripdata_2023-01.parquet"), dataset)
        df_val = read_dataframe(os.path.join(raw_data_path, f"{dataset}_tripdata_2023-02.parquet"), dataset)
        df_test = read_dataframe(os.path.join(raw_data_path, f"{dataset}_tripdata_2023-03.parquet"), dataset)
        
        # Extract target
        target = 'duration'
        y_train = df_train[target].values
        y_val = df_val[target].values
        y_test = df_test[target].values
        
        # Preprocess data
        dv = DictVectorizer(sparse=True)  # Ensure sparse output
        X_train, dv = preprocess(df_train, dv, fit_dv=True)
        X_val, _ = preprocess(df_val, dv, fit_dv=False)
        X_test, _ = preprocess(df_test, dv, fit_dv=False)
        
        # Save results
        os.makedirs(dest_path, exist_ok=True)
        dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
        dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
        dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
        dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))
        
        logging.info(f"Preprocessing complete. Saved 4 files to {dest_path}")
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise

if __name__ == '__main__':
    run_data_prep()
