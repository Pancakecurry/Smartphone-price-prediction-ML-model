"""
Tree-Based ML Pipeline optimized via Optuna Bayesian Search.

Contains the TreeModelTrainer inheriting the baseline MLflow framework
to dynamically identify hyperparameter bounds for RandomForest and XGBoost regressors.
Includes a unified sklearn Pipeline method that bakes preprocessing into the model
artifact to eliminate training/inference scaling mismatch.
"""
import optuna
import mlflow
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from src.models.model_evaluator import BaseModelTrainer
from src.logger import get_logger

logger = get_logger(__name__)


class TreeModelTrainer(BaseModelTrainer):
    """
    Advanced Tree Modeling class mapping rigorous hyperparameter optimizations
    against Scikit-Learn validation boundaries iteratively.
    """

    def __init__(self, experiment_name: str = "Smartphone_Price_Prediction"):
        super().__init__(experiment_name=experiment_name)

    # ------------------------------------------------------------------
    # Legacy: trains on pre-scaled .npy arrays (kept for backward compat)
    # ------------------------------------------------------------------
    def train_random_forest(self, n_trials: int = 20) -> dict:
        """
        Executes Optuna optimization searching for the best Random Forest constraints.
        Logs resulting optimal metrics and algorithms directly to MLflow.
        NOTE: trains on pre-scaled arrays from BaseModelTrainer._load_datasets().
        """
        with mlflow.start_run(run_name="RandomForest_Optuna_Optimized"):
            logger.info(f"Initiating Random Forest Optuna Search ({n_trials} Trials)...")

            def objective(trial: optuna.Trial) -> float:
                n_estimators = trial.suggest_int("n_estimators", 50, 300)
                max_depth = trial.suggest_int("max_depth", 5, 20)

                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1
                )
                scores = cross_val_score(
                    rf, self.X_train, self.y_train,
                    cv=2, scoring="neg_mean_squared_error"
                )
                return float(scores.mean())

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_params
            logger.info(f"Random Forest Best Optuna Params: {best_params}")

            final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
            final_model.fit(self.X_train, self.y_train)

            y_pred = final_model.predict(self.X_test)
            metrics = self.evaluate_model(self.y_test, y_pred)

            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(final_model, "model")

            return metrics

    def train_xgboost(self, n_trials: int = 20) -> dict:
        """
        Executes Optuna bayesian learning rate bounding against Gradient Boosted trees.
        Logs final structure into MLflow directly.
        """
        with mlflow.start_run(run_name="XGBoost_Optuna_Optimized"):
            logger.info(f"Initiating XGBoost Optuna Search ({n_trials} Trials)...")

            def objective(trial: optuna.Trial) -> float:
                learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
                max_depth = trial.suggest_int("max_depth", 3, 15)
                n_estimators = trial.suggest_int("n_estimators", 50, 300)

                model = xgb.XGBRegressor(
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    random_state=42,
                    n_jobs=-1
                )
                scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=2, scoring="neg_mean_squared_error"
                )
                return float(scores.mean())

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_params
            logger.info(f"XGBoost Best Optuna Params: {best_params}")

            final_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
            final_model.fit(self.X_train, self.y_train)

            y_pred = final_model.predict(self.X_test)
            metrics = self.evaluate_model(self.y_test, y_pred)

            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(final_model, "model")

            return metrics

    # ------------------------------------------------------------------
    # ✅ UNIFIED PIPELINE: ColumnTransformer → RandomForest
    # ------------------------------------------------------------------
    def train_random_forest_pipeline(self, n_trials: int = 20) -> dict:
        """
        Builds and trains a UNIFIED sklearn Pipeline: ColumnTransformer → RandomForest.

        Unlike train_random_forest(), this method:
          - Fetches raw (unscaled) X_train / y_train directly from SmartphoneFeatureEngineer
          - Wraps the ColumnTransformer and RandomForestRegressor into a single Pipeline
            so preprocessing is baked into the model artifact
          - Logs the unified pipeline to MLflow under artifact name 'RandomForest_Pipeline'
            so the backend can call pipeline.predict(raw_dataframe) with zero manual scaling

        This eliminates the training/inference preprocessing mismatch entirely.
        """
        from src.features.feature_engineering import SmartphoneFeatureEngineer

        with mlflow.start_run(run_name="RandomForest_Pipeline"):
            logger.info(f"Initiating Unified Pipeline Training ({n_trials} Optuna Trials)...")

            # Fetch raw (unscaled) Pandas DataFrames from the feature engineer
            engineer = SmartphoneFeatureEngineer()
            X_train_raw = engineer.X_train   # Pandas DataFrame, NOT scaled .npy
            X_test_raw  = engineer.X_test
            y_train_raw = engineer.y_train
            y_test_raw  = engineer.y_test
            preprocessor = engineer.preprocessor

            logger.info(f"Raw training data shape: {X_train_raw.shape}")

            # Optuna search over the unified pipeline
            def objective(trial: optuna.Trial) -> float:
                n_estimators = trial.suggest_int("n_estimators", 50, 300)
                max_depth    = trial.suggest_int("max_depth", 5, 20)

                candidate = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=-1
                    ))
                ])
                scores = cross_val_score(
                    candidate, X_train_raw, y_train_raw,
                    cv=2, scoring="neg_mean_squared_error"
                )
                return float(scores.mean())

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_params
            logger.info(f"Unified Pipeline Best Params: {best_params}")

            # Train the final unified pipeline on full raw training data
            unified_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(
                    **best_params, random_state=42, n_jobs=-1
                ))
            ])
            unified_pipeline.fit(X_train_raw, y_train_raw)

            # Evaluate — pipeline handles all scaling internally
            y_pred = unified_pipeline.predict(X_test_raw)
            metrics = self.evaluate_model(y_test_raw.to_numpy(), y_pred)

            logger.info(
                f"Unified Pipeline | RMSE: {metrics['RMSE']:.4f} | "
                f"R²: {metrics['R2_Score']:.4f}"
            )

            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            # Artifact name 'RandomForest_Pipeline' — backend searches for this tag
            mlflow.sklearn.log_model(unified_pipeline, "RandomForest_Pipeline")

            logger.info("✅ Unified sklearn Pipeline logged to MLflow as 'RandomForest_Pipeline'.")
            return metrics


if __name__ == "__main__":
    logger.info("Initializing phase 3 execution modeling constraints.")
    trainer = TreeModelTrainer()

    trainer.train_random_forest(n_trials=3)
    trainer.train_xgboost(n_trials=3)
    trainer.train_random_forest_pipeline(n_trials=3)

    logger.info("Modeling graph fully successfully closed.")
