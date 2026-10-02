from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, output_path):
        pass

    @abstractmethod
    def load(self, model_path):
        pass

    def feature_importance(self):
        return None


class MultitaskMutationNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers, dropout: float, head_layers=None):
        super().__init__()
        layers = []
        prev = input_size
        for hidden in hidden_layers:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        self.feature_extractor = nn.Sequential(*layers)
        head_layers = head_layers or []
        head_modules = []
        head_prev = prev
        for head_hidden in head_layers:
            head_modules.append(nn.Linear(head_prev, head_hidden))
            head_modules.append(nn.LayerNorm(head_hidden))
            head_modules.append(nn.GELU())
            if dropout > 0:
                head_modules.append(nn.Dropout(dropout))
            head_prev = head_hidden
        head_modules.append(nn.Linear(head_prev, output_size))
        self.head = nn.Sequential(*head_modules)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.head(features)

    def get_head_weights(self):
        """Extract the weights of the final linear layer in the head."""
        for module in reversed(self.head):
            if isinstance(module, nn.Linear):
                return module.weight.detach().clone()
        raise ValueError("No Linear layer found in head")

    def get_encoder_output(self, x):
        """Get the encoder/feature extractor output for input samples."""
        with torch.no_grad():
            return self.feature_extractor(x)


class MultitaskMutationModel(BaseModel):
    def __init__(self, input_size: int, output_size: int, **kwargs):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = kwargs.get('hidden_layers', [512, 256, 128])
        self.head_layers = kwargs.get('head_layers', [])
        self.dropout = kwargs.get('dropout_rate', 0.2)
        self.lr = kwargs.get('learning_rate', 5e-4)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.batch_size = kwargs.get('batch_size', 128)
        self.epochs = kwargs.get('epochs', 60)
        self.val_split = kwargs.get('validation_split', 0.15)
        self.patience = kwargs.get('patience', 10)
        self.gradient_clip = kwargs.get('gradient_clip', 1.0)
        self.use_pos_weight = kwargs.get('use_pos_weight', True)
        self.normalize_inputs = kwargs.get('normalize_inputs', True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = []
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler: StandardScaler | None = None

    def _build_network(self):
        return MultitaskMutationNet(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            head_layers=self.head_layers
        )

    def _prepare_dataloaders(self, X: np.ndarray, y: np.ndarray):
        from sklearn.model_selection import train_test_split

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if not np.isfinite(X).all():
            raise ValueError("Input features contain NaNs or infs. Please clean the data before training.")
        if not np.isfinite(y).all():
            raise ValueError("Target matrix contains NaNs or infs.")

        if self.normalize_inputs:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X).astype(np.float32)
        else:
            self.scaler = None

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.val_split,
            random_state=42,
            shuffle=True
        )

        X_train_tensor = torch.from_numpy(X_train)
        y_train_tensor = torch.from_numpy(y_train)
        X_val_tensor = torch.from_numpy(X_val)
        y_val_tensor = torch.from_numpy(y_val)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        if self.use_pos_weight:
            pos_counts = y_train_tensor.sum(dim=0)
            neg_counts = y_train_tensor.shape[0] - pos_counts
            pos_weight = torch.where(pos_counts > 0, neg_counts / pos_counts, torch.ones_like(pos_counts))
            pos_weight = torch.clamp(pos_weight, min=1.0, max=1e6)
            pos_weight = pos_weight.to(self.device)
        else:
            pos_weight = None

        return train_loader, val_loader, pos_weight

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        train_loader, val_loader, pos_weight = self._prepare_dataloaders(X, y)
        self.model = self._build_network().to(self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=max(1, self.patience // 3),
        )

        best_val = float('inf')
        epochs_no_improve = 0
        self.training_history = []
        best_state = None

        for epoch in range(self.epochs):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            scheduler.step(val_loss)

            self.training_history.append(
                {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': self.optimizer.param_groups[0]['lr']
                }
            )

            if val_loss + 1e-5 < best_val or best_state is None:
                best_val = val_loss
                epochs_no_improve = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def fit_full_dataset(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame):
        """Train the model on the full dataset without validation split."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if not np.isfinite(X).all():
            raise ValueError("Input features contain NaNs or infs. Please clean the data before training.")
        if not np.isfinite(y).all():
            raise ValueError("Target matrix contains NaNs or infs.")

        if self.normalize_inputs:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X).astype(np.float32)
        else:
            self.scaler = None

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        train_loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        if self.use_pos_weight:
            pos_counts = y_tensor.sum(dim=0)
            neg_counts = y_tensor.shape[0] - pos_counts
            pos_weight = torch.where(pos_counts > 0, neg_counts / pos_counts, torch.ones_like(pos_counts))
            pos_weight = torch.clamp(pos_weight, min=1.0, max=1e6)
            pos_weight = pos_weight.to(self.device)
        else:
            pos_weight = None

        self.model = self._build_network().to(self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

        self.training_history = []
        for epoch in range(self.epochs):
            train_loss = self._run_epoch(train_loader, train=True)
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            })

    def get_head_weights(self):
        """Extract the weights of the final linear layer in the head."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() or fit_full_dataset() first.")
        return self.model.get_head_weights()

    def get_sample_embeddings(self, X: np.ndarray | pd.DataFrame):
        """Get encoder/feature extractor embeddings for input samples."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() or fit_full_dataset() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.asarray(X, dtype=np.float32)
        if self.normalize_inputs and self.scaler is not None:
            X = self.scaler.transform(X).astype(np.float32)

        X_tensor = torch.from_numpy(X).to(self.device)
        return self.model.get_encoder_output(X_tensor)

    def _run_epoch(self, loader: DataLoader, train: bool = True) -> float:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0
        with torch.set_grad_enabled(train):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)

                if train:
                    loss.backward()
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()

                total_loss += loss.item() * batch_X.size(0)
                total_samples += batch_X.size(0)

        return total_loss / max(total_samples, 1)

    def predict(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        if self.normalize_inputs and self.scaler is not None:
            X = self.scaler.transform(X).astype(np.float32)
        X_tensor = torch.from_numpy(X)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_logits = []
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                logits = self.model(batch_X)
                all_logits.append(logits.cpu())

        logits = torch.cat(all_logits, dim=0)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int)
        return preds, probs

    def save(self, output_path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model_state': self.model.state_dict(),
            'config': {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'hidden_layers': self.hidden_layers,
                'head_layers': self.head_layers,
                'dropout': self.dropout
            },
            'training_history': self.training_history,
            'scaler': None
        }
        if self.normalize_inputs and self.scaler is not None:
            checkpoint['scaler'] = {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist(),
                'var': getattr(self.scaler, 'var_', np.square(self.scaler.scale_)).tolist(),
                'n_features_in': getattr(self.scaler, 'n_features_in_', len(self.scaler.scale_))
            }
        torch.save(checkpoint, output_path / 'multitask_nn.pt')
        torch.save(self.model.feature_extractor.state_dict(), output_path / 'encoder.pt')

        if self.training_history:
            history_df = pd.DataFrame(self.training_history)
            history_df.to_csv(output_path / 'training_curve.csv', index=False)

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = self._build_network().to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.training_history = checkpoint.get('training_history', [])
        scaler_state = checkpoint.get('scaler')
        if scaler_state:
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(scaler_state['mean'], dtype=np.float32)
            self.scaler.scale_ = np.array(scaler_state['scale'], dtype=np.float32)
            self.scaler.var_ = np.array(scaler_state.get('var', np.square(self.scaler.scale_)), dtype=np.float32)
            self.scaler.n_features_in_ = scaler_state.get('n_features_in', self.scaler.mean_.shape[0])
        else:
            self.scaler = None
        self.model.eval()

    def feature_importance(self):
        return None


class ModelFactory:
    def get_model(self, model_name: str, input_size: int, output_size: int, config: dict | None = None):
        config = config or {}
        model_section = config.get('model', {}) or {}
        model_config = model_section.get(model_name) or {}

        if model_name == 'multitask_nn':
            return MultitaskMutationModel(input_size=input_size, output_size=output_size, **model_config)

        raise ValueError(f"Unsupported model type: {model_name}. Available options: multitask_nn")
