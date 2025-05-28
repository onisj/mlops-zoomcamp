import os
import pickle
import logging
import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_pickle(filename: str):
    """Load pickle file."""
    try:
        with open(filename, "rb") as f_in:
            return pickle.load(f_in)
        logging.info(f"Loaded {filename}")
    except Exception as e:
        logging.error(f"Failed to load {filename}: {e}")
        raise

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    """Train RandomForestRegressor and log to MLflow."""
    try:
        # Set MLflow tracking
        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        mlflow.set_experiment("random-forest-train")
        mlflow.sklearn.autolog()  # Enable autologging for Q3
        
        # Load data
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        
        # Train model
        logging.info("Starting model training")
        with mlflow.start_run():
            rf = RandomForestRegressor(max_depth=10, random_state=0)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            
            # Compute RMSE
            rmse = root_mean_squared_error(y_val, y_pred)
            logging.info(f"Validation RMSE: {rmse:.3f}")
            print(f"Validation RMSE: {rmse:.3f}")
        
        logging.info("Training complete")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == '__main__':
    run_train()
