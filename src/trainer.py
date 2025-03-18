from src.preprocess.preprocess import Preprocessor
from src.config import Config
from src.model import init_model
from src.logger import logger
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np
from typing import Dict, Tuple
import os
from torch.utils.data import TensorDataset, DataLoader


class Trainer:
    def __init__(self, config_path: str, csv_file: str, device: str = "cuda"):
        self.config = Config.from_file(config_path)
        os.makedirs(self.config.log_dir, exist_ok=True)
        self.proc = Preprocessor(csv_file, self.config.target_col_name, self.config.ignore_col_names)
        self.model = None
        self.feature_list = None
        self.scalers = None
        self.process()
        self.device = device

    def process(self):
        self.proc.fill_null(self.config.preprocess_params.fill_na)
        self.proc.treat_outliers(self.config.preprocess_params.iqr_factor, self.config.preprocess_params.iqr_limits[0],
                                 self.config.preprocess_params.iqr_limits[1],
                                 self.config.preprocess_params.fill_outlier,
                                 self.config.preprocess_params.target_outlier)
        self.feature_list, self.scalers = self.proc.get_feature_list_and_scalers(
            self.config.preprocess_params.n_components)
        logger.info("Using {} features for training".format(len(self.feature_list)))

    def _init_model(self, input_dim: int):
        self.model = init_model(self.config.model_params.name, input_dim, self.device,
                                self.config.model_params.params)

    def _prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.proc.df[self.feature_list]
        target_col = self.proc.df.columns[-1]
        y = self.proc.df[target_col].values
        X_scaled = self.scalers.feature_scaler.transform(X)
        y_reshaped = y.reshape(-1, 1)
        y_scaled = self.scalers.target_scaler.transform(y_reshaped)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)
        return X_tensor, y_tensor

    def train(self) -> Dict:
        self._init_model(len(self.feature_list))
        X, y = self._prepare_data()
        criterion = nn.MSELoss()
        kf = KFold(n_splits=self.config.train_params.n_splits, shuffle=True, random_state=42)

        fold_results = []
        best_model = None
        best_val_loss = float('inf')

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"Training fold {fold + 1}/{self.config.train_params.n_splits}")
            self._init_model(len(self.feature_list))

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=self.config.train_params.batch_size, shuffle=True)

            optimizer = optim.Adam(self.model.parameters(), lr=self.config.train_params.lr)
            train_losses = []
            val_losses = []

            for epoch in range(self.config.train_params.epochs):
                self.model.train()
                batch_losses = []

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())

                epoch_loss = np.mean(batch_losses)

                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val)

                logger.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{self.config.train_params.epochs}, "
                            f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss.item():.4f}")

                train_losses.append(epoch_loss)
                val_losses.append(val_loss.item())

            final_train_loss = train_losses[-1]
            final_val_loss = val_losses[-1]

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_model = {"state_dict": self.model.state_dict().copy(), "scalers": self.scalers.to_dict(),
                              "feature_list": self.feature_list}

            fold_result = {
                "fold": fold + 1,
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
                "train_loss_history": train_losses,
                "val_loss_history": val_losses
            }
            fold_results.append(fold_result)

            logger.info(
                f"Fold {fold + 1} completed. Train Loss: {final_train_loss:.4f}, Val Loss: {final_val_loss:.4f}")

        avg_train_loss = np.mean([fold["train_loss"] for fold in fold_results])
        avg_val_loss = np.mean([fold["val_loss"] for fold in fold_results])

        logger.info(f"Cross-validation completed.")
        logger.info(f"Average Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")

        if best_model is not None:
            self.model.load_state_dict(best_model["state_dict"])
            logger.info("Loaded best model from cross-validation")

        model_path = os.path.join(self.config.log_dir, "best_model.pth")
        torch.save(best_model, model_path)
        logger.info(f"Best model saved to {model_path}")

        results = {
            "fold_results": fold_results,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "best_val_loss": best_val_loss,
            "model_path": model_path
        }

        return results



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train model on a config and csv file")
    parser.add_argument("--config_file", type=str, help="Path to config file")
    parser.add_argument("--input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("--device", type=str, help="device name cuda/cpu/mps")

    args = parser.parse_args()

    trainer = Trainer(args.config_file, args.input_file)
    trainer.train()
