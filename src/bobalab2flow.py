# src/bobalab2flow.py

from metaflow import FlowSpec, step, Parameter
import mlflow
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class BobaLab2Flow(FlowSpec):
    """
    Metaflow flow for Lab 2:
    - Ingest Yelp JSON for reviews & business
    - Engineer features around opening hours
    - Train Decision Tree & Logistic Regression
    - Log & register final logistic model in MLflow
    """

    # allow overriding max tree depth & test size
    max_depth     = Parameter("max_depth", default=4, help="max_depth for DecisionTree")
    test_size     = Parameter("test_size", default=0.2, help="fraction for test split")
    random_seed   = Parameter("seed", default=42, help="random seed")
    mlflow_db_uri = Parameter("mlflow_db_uri", default="sqlite:///mlflow.db", help="MLflow tracking URI")

    @step
    def start(self):
        """Load raw Yelp JSON files into DataFrames"""
        # adjust paths as needed
        business = pd.read_json(
            "/Users/rebekahzhou/Downloads/Yelp JSON/yelp_dataset/"
            "yelp_academic_dataset_business.json",
            lines=True
        )
        # we'll predict business.is_open
        self.boba = business.copy()
        self.next(self.transform)

    @step
    def transform(self):
        """Feature‐engineer opening hours"""
        boba = self.boba

        # helper to get latest closing hour
        def latest_closing_time(hours_dict):
            latest = -1
            if not isinstance(hours_dict, dict):
                return None
            for day_times in hours_dict.values():
                for time in day_times.split(","):
                    try:
                        # “HH:MM-HH:MM”
                        close_hour, close_min = map(int, time.split("-")[1].split(":"))
                        minutes = close_hour*60 + close_min
                        if minutes < 5*60:  # treat e.g. 2AM as next day
                            minutes += 24*60
                        latest = max(latest, minutes)
                    except:
                        continue
            return (latest / 60) if latest >= 0 else None

        boba["latest_close_hour"] = boba["hours"].apply(latest_closing_time).fillna(0)

        # open all 7 days?
        boba["open_7_days"] = boba["hours"].apply(
            lambda h: int(isinstance(h, dict) and len(h)==7)
        )

        # open weekend?
        def is_open_weekends(h):
            if not isinstance(h, dict):
                return 0
            return int("Saturday" in h and "Sunday" in h)
        boba["open_weekends"] = boba["hours"].apply(is_open_weekends)

        # open each weekday
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        for d in days:
            boba[f"open_{d.lower()}"] = boba["hours"].apply(
                lambda h, d=d: int(isinstance(h, dict) and d in h)
            )

        # finalize feature set
        numerical = ["latest_close_hour"]
        categorical = ["open_7_days","open_weekends"] + [f"open_{d.lower()}" for d in days]

        # store
        self.X   = boba[numerical + categorical].astype(float)
        self.y   = boba["is_open"]
        self.next(self.split)

    @step
    def split(self):
        """Train/test split"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_seed
        )
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.next(self.train)

    @step
    def train(self):
        """Train two models and log to MLflow"""
        mlflow.set_tracking_uri(self.mlflow_db_uri)
        mlflow.set_experiment("yelp-lab-2")

        final_models = {}
        best_params  = {}

        # Decision Tree
        with mlflow.start_run(run_name="decision_tree"):
            mlflow.set_tag("Model","decision-tree")
            mlflow.log_param("max_depth", self.max_depth)
            mlflow.log_param("seed", self.random_seed)

            dt = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_seed
            )
            dt.fit(self.X_train, self.y_train)
            acc = accuracy_score(self.y_test, dt.predict(self.X_test))
            mlflow.log_metric("test_accuracy", acc)

            mlflow.sklearn.log_model(
                sk_model=dt,
                artifact_path="decision_tree_model"
            )
            final_models["dt"] = dt
            best_params["dt"] = {"max_depth": self.max_depth}

        # Logistic Regression
        with mlflow.start_run(run_name="logreg") as run:
            mlflow.set_tag("Model","logistic-regression")
            lr = LogisticRegression(max_iter=200)
            lr.fit(self.X_train, self.y_train)
            acc = accuracy_score(self.y_test, lr.predict(self.X_test))
            mlflow.log_metric("test_accuracy", acc)

            mlflow.sklearn.log_model(
                sk_model=lr,
                artifact_path="logreg_model"
            )
            final_models["logreg"] = lr
            best_params["logreg"] = {"max_iter": 200}

            # capture run_id inside the with-block
            run_id = run.info.run_id

        # Register the logistic model
        uri = f"runs:/{run_id}/logreg_model"
        mlflow.register_model(model_uri=uri, name="boba_logreg")

        self.final_models = final_models
        self.next(self.end)

    @step
    def end(self):
        print("✅ All runs complete. Your `boba_logreg` model is now registered in MLflow.")
        for name, mdl in self.final_models.items():
            print(f"- {name} trained; check its run in the MLflow UI.")

if __name__ == "__main__":
    BobaLab2Flow()
