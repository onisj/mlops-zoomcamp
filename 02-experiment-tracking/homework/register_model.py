import os
import pickle
import logging
import click
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import root_mean_squared_error

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5001")

def load_pickle(filename):
    """Load pickle data."""
    try:
        with open(filename, "rb") as f_in:
            logging.info(f"Loaded {filename}")
            return pickle.load(f_in)
    except Exception as e:
        logging.error(f"Error loading {filename}: {e}")
        raise

def ensure_experiment(client: MlflowClient, experiment_name: str):
    """Ensure MLflow experiment exists, create if not."""
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name)
            logging.info(f"Created experiment {experiment_name} with ID {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            logging.info(f"Found experiment {experiment_name} with ID {experiment_id}")
        return experiment_id
    except Exception as e:
        logging.error(f"Failed to ensure experiment {experiment_name}: {e}")
        raise

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models to evaluate"
)
def run_register_model(data_path: str, top_n: int):
    """Evaluate top models and register the best one."""
    try:
        client = MlflowClient()
        
        # Ensure experiments exist
        hpo_exp_id = ensure_experiment(client, HPO_EXPERIMENT_NAME)
        best_exp_id = ensure_experiment(client, EXPERIMENT_NAME)
        
        # Retrieve top_n model runs
        logging.info(f"Retrieving top {top_n} runs from {HPO_EXPERIMENT_NAME}")
        runs = client.search_runs(
            experiment_ids=[hpo_exp_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=top_n,
            order_by=["metrics.rmse ASC"]
        )
        if not runs:
            logging.error(f"No runs found in {HPO_EXPERIMENT_NAME}")
            raise ValueError(f"No runs found in {HPO_EXPERIMENT_NAME}")
        
        # Load test data
        X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))
        
        best_rmse = float('inf')
        best_run_id = None
        
        # Evaluate each run on test set
        for run in runs:
            try:
                run_id = run.info.run_id
                model_uri = f"runs:/{run_id}/model"
                logging.info(f"Evaluating run {run_id}")
                
                # Load and evaluate model
                model = mlflow.sklearn.load_model(model_uri)
                y_pred = model.predict(X_test)
                rmse = root_mean_squared_error(y_test, y_pred)
                
                # Log to new experiment
                with mlflow.start_run(experiment_id=best_exp_id):
                    mlflow.log_metric("test_rmse", rmse)
                    mlflow.log_param("source_run_id", run_id)
                    logging.info(f"Logged test_rmse={rmse:.3f} for run {run_id}")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_run_id = run_id
            except Exception as e:
                logging.error(f"Failed to evaluate run {run_id}: {e}")
                continue
        
        if best_run_id is None:
            logging.error("No valid models evaluated")
            raise ValueError("No valid models evaluated")
        
        # Register the best model
        model_uri = f"runs:/{best_run_id}/model"
        model_name = "nyc-taxi-best-model"
        logging.info(f"Registering model from run {best_run_id} as {model_name}")
        mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"Registered model with test RMSE: {best_rmse:.3f}")
        
        logging.info("Model registration complete")
    except Exception as e:
        logging.error(f"Model registration failed: {e}")
        raise

if __name__ == '__main__':
    run_register_model()
