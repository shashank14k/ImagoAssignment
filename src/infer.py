import os
import torch
from typing import Dict
import pandas as pd
from src.config import Config
from src.model import init_model
from src.preprocess.preprocess import DataScaler
from src.logger import logger

def validate_ckp(ckp: Dict):
    assert "scalers" in ckp.keys(), "Missing key scaler in checkpoint"
    assert "feature_list" in ckp.keys(), "Missing key feature_list in checkpoint"
    assert "state_dict" in ckp.keys(), "Missing key state_dict in checkpoint"

class Predictor:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        logger.info("Initializing Predictor...")
        assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist."
        assert os.path.exists(config_path), f"Config path {config_path} does not exist."
        self.config = Config.from_file(config_path)
        self.ckp = torch.load(checkpoint_path, weights_only=False)
        validate_ckp(self.ckp)
        self.scalers = DataScaler(**self.ckp["scalers"])
        self.feature_list = self.ckp["feature_list"]
        logger.info(f"Loaded feature list: {self.feature_list}")
        self.model = init_model(self.config.model_params.name, len(self.feature_list), device,
                                self.config.model_params.params)
        self.model.load_state_dict(self.ckp["state_dict"])
        self.model = self.model.to(device).eval()
        self.device = device
        logger.info("Predictor initialized successfully.")

    def df_to_feature_tensor(self, df: pd.DataFrame):
        try:
            df = df[self.feature_list]
            logger.debug("Transforming features using feature scaler...")
            X_scaled = self.scalers.feature_scaler.transform(df)
            X_scaled = torch.Tensor(X_scaled).to(self.device)
            return X_scaled
        except Exception as e:
            logger.error("Error in df_to_feature_tensor: {}".format(e))
            raise ValueError("Missing features detected")

    def infer(self, x: Dict):
        missing_features = set(self.feature_list) - set(x.keys())
        if missing_features:
            error_msg = f"Missing required features: {missing_features}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Inferring prediction for input: {x}")
        df = pd.DataFrame(x, index=[0])
        X_tensor = self.df_to_feature_tensor(df)
        with torch.no_grad():
            preds = self.model(X_tensor)
            preds = preds.to("cpu").numpy()
        prediction = self.scalers.target_scaler.inverse_transform(preds)[0]
        logger.info(f"Prediction result: {prediction}")
        return prediction

    def batch_infer(self, df: pd.DataFrame):
        logger.info("Performing batch inference...")
        missing_features = set(df.columns) - set(self.feature_list)
        if missing_features:
            error_msg = f"DataFrame is missing required features: {missing_features}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        X_tensor = self.df_to_feature_tensor(df)
        with torch.no_grad():
            preds = self.model(X_tensor)
            preds = preds.to("cpu").numpy()
        predictions = self.scalers.target_scaler.inverse_transform(preds)
        logger.info("Batch inference completed.")
        return predictions