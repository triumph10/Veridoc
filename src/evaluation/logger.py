import mlflow

def log_experiment(run_name: str, params: dict, metrics: dict):
    mlflow.set_tracking_uri("sqlite:///mlflow.db") # add this line
    mlflow.set_experiment("veridoc_eval")
    
    with mlflow.start_run(run_name=run_name):
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, float(value))
            except (ValueError, TypeError):
                print(f"Skipping metric {key} — value was NaN")
    
    print(f"Run '{run_name}' logged to MLflow.")