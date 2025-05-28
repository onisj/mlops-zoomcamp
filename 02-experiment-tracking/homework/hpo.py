import os
import pickle
import logging
import click
import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option("--data_path", default="./output")
@click.option("--num_trials", default=5, type=int)
def run_optimization(data_path, num_trials):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("random-forest-hyperopt")

    def objective(params):
        logging.info(f"Starting trial with params: {params}")
        with mlflow.start_run():
            logging.info("Started MLflow run")
            mlflow.sklearn.autolog()
            rf = RandomForestRegressor(**params, n_jobs=1)
            logging.info("Fitting model")
            rf.fit(X_train, y_train)
            logging.info("Predicting on validation set")
            y_pred = rf.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            logging.info(f"Validation RMSE: {rmse:.3f}")
            return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': hp.choice('max_depth', range(1, 10)),
        'n_estimators': hp.choice('n_estimators', range(10, 30)),
        'min_samples_split': hp.choice('min_samples_split', range(2, 10)),
        'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 5)),
        'random_state': 42
    }

    logging.info(f"Starting hyperparameter optimization with {num_trials} trials")
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials()
    )
    logging.info(f"Best params: {best_result}")

if __name__ == '__main__':
    run_optimization()
