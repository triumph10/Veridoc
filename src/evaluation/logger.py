import mlflow


def log_experiment(run_name: str, params: dict, metrics: dict):
    """Log parameters and metrics to MLflow."""
    mlflow.set_experiment("veridoc_eval")
    
    with mlflow.start_run(run_name=run_name):
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))
    
    print(f"Run '{run_name}' logged to MLflow.")