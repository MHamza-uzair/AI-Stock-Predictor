"""
Training loop for short-term and long-term LSTM models.
Handles early stopping, LR scheduling, gradient clipping, and checkpoint saving.
"""
import json
import logging
import os
import random
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.ensemble import train_xgb
from src.feature_engineering import (
    apply_scaler,
    build_sequences_seq2seq,
    chronological_split,
    compute_features,
    fit_scalers,
    load_scalers,
    FEATURE_COLUMNS,
)
from src.model import (
    StockLSTM,
    build_long_term_model,
    build_short_term_model,
    count_parameters,
    ST_LOOKBACK,
    ST_PREDICTION_HORIZON,
    LT_LOOKBACK,
    LT_PREDICTION_HORIZON,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLOSE_COL_IDX = FEATURE_COLUMNS.index("Close")  # Index of Close in feature matrix

# Short-term training hyperparameters
ST_LR = 0.001
ST_BATCH_SIZE = 32
ST_MAX_EPOCHS = 100
ST_EARLY_STOP_PATIENCE = 15
ST_LR_PATIENCE = 5
ST_LR_FACTOR = 0.5

# Long-term training hyperparameters
LT_LR = 0.0005
LT_BATCH_SIZE = 16
LT_MAX_EPOCHS = 150
LT_EARLY_STOP_PATIENCE = 20
LT_LR_PATIENCE = 7
LT_LR_FACTOR = 0.5

GRAD_CLIP_NORM = 1.0        # Max L2 norm for gradient clipping
RANDOM_SEED = 42


def _set_seeds() -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _model_path(ticker: str, model_type: str) -> str:
    """Return the filesystem path for a model checkpoint."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, f"{ticker}_{model_type}_model.pt")


def _metrics_path(ticker: str) -> str:
    """Return the path for the metrics JSON file."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, f"{ticker}_metrics.json")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _make_tensors(
    X: np.ndarray, y: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert numpy arrays to float32 PyTorch tensors.

    y is already (N, horizon) from build_sequences_seq2seq — no unsqueeze needed.
    """
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),  # shape: (N, horizon)
    )


def _build_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Wrap numpy arrays in PyTorch DataLoaders."""
    Xt, yt = _make_tensors(X_train, y_train)
    Xv, yv = _make_tensors(X_val, y_val)

    train_loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=batch_size,
        shuffle=True,       # Shuffle within each epoch to reduce sequential bias
    )
    val_loader = DataLoader(
        TensorDataset(Xv, yv),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def _run_epoch(
    model: StockLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    is_train: bool,
) -> float:
    """
    Run one epoch of training or validation.

    Parameters
    ----------
    model : StockLSTM
    loader : DataLoader
    criterion : loss function
    optimizer : Adam optimizer (None during validation)
    is_train : bool
        If True, compute gradients and update weights.

    Returns
    -------
    float
        Mean loss over all batches in this epoch.
    """
    model.train(is_train)
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping prevents exploding gradients in deep LSTMs
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def _train_model(
    model: StockLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    max_epochs: int,
    early_stop_patience: int,
    lr_patience: int,
    lr_factor: float,
    checkpoint_path: str,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> Dict[str, float]:
    """
    Full training loop with early stopping, LR scheduling, and checkpointing.

    Parameters
    ----------
    model : StockLSTM
    train_loader, val_loader : DataLoaders
    lr : float
        Initial learning rate.
    max_epochs : int
    early_stop_patience : int
        Stop training if val loss doesn't improve for this many epochs.
    lr_patience, lr_factor : int, float
        ReduceLROnPlateau parameters.
    checkpoint_path : str
        Where to save the best model weights.
    progress_callback : callable, optional
        Called with (epoch, max_epochs, train_loss, val_loss) each epoch.
        Used by the Streamlit UI to update a progress bar.

    Returns
    -------
    Dict[str, float]
        Final best train/val losses.
    """
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=lr_patience, factor=lr_factor,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, is_train=True)
        val_loss = _run_epoch(model, val_loader, criterion, None, is_train=False)

        scheduler.step(val_loss)

        logger.info(
            "Epoch %d/%d — train_loss=%.6f  val_loss=%.6f  lr=%.6f",
            epoch, max_epochs, train_loss, val_loss,
            optimizer.param_groups[0]["lr"],
        )

        if progress_callback:
            progress_callback(epoch, max_epochs, train_loss, val_loss)

        # Save checkpoint whenever validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            try:
                torch.save(model.state_dict(), checkpoint_path)
                logger.info("Checkpoint saved (val_loss=%.6f) at %s", best_val_loss, checkpoint_path)
            except Exception as exc:
                logger.error("Failed to save checkpoint: %s", exc)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    early_stop_patience,
                )
                break

    return {"best_val_loss": best_val_loss, "final_train_loss": train_loss}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_on_test(
    model: StockLSTM,
    X_test: np.ndarray,
    y_test: np.ndarray,
    close_scaler,
    dir_step: int = 5,
) -> Dict[str, float]:
    """
    Run the model on the held-out test set and compute evaluation metrics.

    With seq2seq output the model returns (n_samples, horizon) predictions.
    RMSE/MAE/MAPE are computed over all (sample, step) pairs flattened together.
    Directional accuracy uses an N-day horizon comparison:
      sign(pred[t+N] - pred[t]) == sign(actual[t+N] - actual[t])
    where t is step 0 and t+N is step dir_step-1 within each test sequence.

    Parameters
    ----------
    model : StockLSTM
    X_test : np.ndarray  Shape (n_samples, lookback, n_features)
    y_test : np.ndarray  Shape (n_samples, horizon) — normalised close prices
    close_scaler : MinMaxScaler  Used to inverse-transform to dollar values
    dir_step : int
        Number of days ahead used for directional accuracy.
        Short-term model uses 5 (weekly); long-term uses 21 (monthly).

    Returns
    -------
    Dict[str, float]
        rmse, mae, mape, directional_accuracy
    """
    model.eval()
    model.to(DEVICE)

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds_norm = model(X_tensor).cpu().numpy()   # (n_samples, horizon)

    n_samples, horizon = preds_norm.shape

    # Flatten all (sample, step) pairs for scalar metrics
    preds_flat = preds_norm.flatten()                # (n_samples * horizon,)
    actual_flat = y_test.flatten()                   # (n_samples * horizon,)

    preds_real = close_scaler.inverse_transform(preds_flat.reshape(-1, 1)).flatten()
    actual_real = close_scaler.inverse_transform(actual_flat.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(np.mean((preds_real - actual_real) ** 2)))
    mae = float(np.mean(np.abs(preds_real - actual_real)))
    mape = float(np.mean(np.abs((preds_real - actual_real) / (actual_real + 1e-8))) * 100)

    # N-day directional accuracy: for each test sequence compare the direction
    # from step 0 to step dir_step against the same span in actuals.
    preds_by_sample  = preds_real.reshape(n_samples, horizon)
    actual_by_sample = actual_real.reshape(n_samples, horizon)
    step = min(dir_step, horizon - 1)   # clamp if horizon is unexpectedly small
    pred_dir   = (preds_by_sample[:, step]  - preds_by_sample[:, 0])  > 0
    actual_dir = (actual_by_sample[:, step] - actual_by_sample[:, 0]) > 0
    dir_acc = float(np.mean(pred_dir == actual_dir) * 100)

    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 4),
        "directional_accuracy": round(dir_acc, 2),
    }
    logger.info("Test metrics (dir_step=%d): %s", dir_step, metrics)
    return metrics


def _save_metrics(all_metrics: Dict, ticker: str) -> None:
    """Persist training metrics to a JSON file in the models directory."""
    path = _metrics_path(ticker)
    try:
        with open(path, "w") as fh:
            json.dump(all_metrics, fh, indent=2)
        logger.info("Metrics saved to %s", path)
    except Exception as exc:
        logger.error("Failed to save metrics: %s", exc)


def load_metrics(ticker: str) -> Optional[Dict]:
    """Load saved metrics for a ticker, or return None if not found."""
    path = _metrics_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.error("Failed to load metrics from %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Public training entry points
# ---------------------------------------------------------------------------

def train_models(
    ticker: str,
    feature_df,           # pd.DataFrame from compute_features()
    progress_callback: Optional[Callable[[str, int, int, float, float], None]] = None,
) -> Dict[str, Dict]:
    """
    Train both the short-term and long-term LSTM models for a given ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (used for file naming).
    feature_df : pd.DataFrame
        Full feature matrix from feature_engineering.compute_features().
    progress_callback : callable, optional
        Called with (model_name, epoch, max_epochs, train_loss, val_loss)
        at each epoch. Used by the Streamlit UI progress bar.

    Returns
    -------
    Dict[str, Dict]
        Nested dict with 'short_term' and 'long_term' sub-dicts each containing
        training results and test metrics.
    """
    _set_seeds()
    logger.info("Starting training pipeline for %s on device: %s", ticker, DEVICE)

    # --- Split and scale ---
    train_end, val_end = chronological_split(feature_df)
    feature_scaler, close_scaler = fit_scalers(feature_df, train_end, ticker)
    scaled_all = apply_scaler(feature_df, feature_scaler)

    all_metrics = {}

    # ---- SHORT-TERM MODEL ----
    logger.info("=== Training Short-Term Model (lookback=%d, horizon=%d) ===",
                ST_LOOKBACK, ST_PREDICTION_HORIZON)
    X_all_st, y_all_st = build_sequences_seq2seq(
        scaled_all, ST_LOOKBACK, ST_PREDICTION_HORIZON, CLOSE_COL_IDX
    )

    # Offset both lookback AND horizon so no y-window crosses a split boundary.
    # Sequence k has y covering rows [k+lookback .. k+lookback+horizon-1].
    # To keep y entirely in training data: k < train_end - lookback - horizon.
    st_train_end = train_end - ST_LOOKBACK - ST_PREDICTION_HORIZON
    st_val_end = val_end - ST_LOOKBACK - ST_PREDICTION_HORIZON

    X_train_st, y_train_st = X_all_st[:st_train_end], y_all_st[:st_train_end]
    X_val_st, y_val_st = X_all_st[st_train_end:st_val_end], y_all_st[st_train_end:st_val_end]
    X_test_st, y_test_st = X_all_st[st_val_end:], y_all_st[st_val_end:]

    logger.info(
        "ST sequences — train: %d, val: %d, test: %d",
        len(X_train_st), len(X_val_st), len(X_test_st)
    )

    st_model = build_short_term_model()
    logger.info("Short-term model parameters: %d", count_parameters(st_model))

    def st_cb(ep, mx, tl, vl):
        if progress_callback:
            progress_callback("short_term", ep, mx, tl, vl)

    train_loader_st, val_loader_st = _build_loaders(
        X_train_st, y_train_st, X_val_st, y_val_st, ST_BATCH_SIZE
    )
    st_train_results = _train_model(
        st_model, train_loader_st, val_loader_st,
        lr=ST_LR, max_epochs=ST_MAX_EPOCHS,
        early_stop_patience=ST_EARLY_STOP_PATIENCE,
        lr_patience=ST_LR_PATIENCE, lr_factor=ST_LR_FACTOR,
        checkpoint_path=_model_path(ticker, "short_term"),
        progress_callback=st_cb,
    )

    # Load the best checkpoint before evaluation
    st_model.load_state_dict(torch.load(_model_path(ticker, "short_term"), map_location=DEVICE))
    st_metrics = evaluate_on_test(st_model, X_test_st, y_test_st, close_scaler, dir_step=7)
    all_metrics["short_term"] = {**st_train_results, **st_metrics}

    # ---- LONG-TERM MODEL ----
    logger.info("=== Training Long-Term Model (lookback=%d, horizon=%d) ===",
                LT_LOOKBACK, LT_PREDICTION_HORIZON)
    X_all_lt, y_all_lt = build_sequences_seq2seq(
        scaled_all, LT_LOOKBACK, LT_PREDICTION_HORIZON, CLOSE_COL_IDX
    )

    lt_train_end = train_end - LT_LOOKBACK - LT_PREDICTION_HORIZON
    lt_val_end = val_end - LT_LOOKBACK - LT_PREDICTION_HORIZON

    # Guard: ensure we have enough sequences for each split
    lt_train_end = max(lt_train_end, 0)
    lt_val_end = max(lt_val_end, lt_train_end)

    X_train_lt = X_all_lt[:lt_train_end]
    y_train_lt = y_all_lt[:lt_train_end]
    X_val_lt = X_all_lt[lt_train_end:lt_val_end]
    y_val_lt = y_all_lt[lt_train_end:lt_val_end]
    X_test_lt = X_all_lt[lt_val_end:]
    y_test_lt = y_all_lt[lt_val_end:]

    logger.info(
        "LT sequences — train: %d, val: %d, test: %d",
        len(X_train_lt), len(X_val_lt), len(X_test_lt)
    )

    if len(X_train_lt) < 1:
        logger.warning("Not enough data for long-term training. Skipping long-term model.")
        all_metrics["long_term"] = {"error": "Insufficient data for long-term model"}
    else:
        lt_model = build_long_term_model()
        logger.info("Long-term model parameters: %d", count_parameters(lt_model))

        def lt_cb(ep, mx, tl, vl):
            if progress_callback:
                progress_callback("long_term", ep, mx, tl, vl)

        train_loader_lt, val_loader_lt = _build_loaders(
            X_train_lt, y_train_lt, X_val_lt, y_val_lt, LT_BATCH_SIZE
        )
        lt_train_results = _train_model(
            lt_model, train_loader_lt, val_loader_lt,
            lr=LT_LR, max_epochs=LT_MAX_EPOCHS,
            early_stop_patience=LT_EARLY_STOP_PATIENCE,
            lr_patience=LT_LR_PATIENCE, lr_factor=LT_LR_FACTOR,
            checkpoint_path=_model_path(ticker, "long_term"),
            progress_callback=lt_cb,
        )

        lt_model.load_state_dict(torch.load(_model_path(ticker, "long_term"), map_location=DEVICE))
        lt_metrics = evaluate_on_test(lt_model, X_test_lt, y_test_lt, close_scaler, dir_step=7)
        all_metrics["long_term"] = {**lt_train_results, **lt_metrics}

    # Train XGBoost direction classifier on the unscaled feature_df
    try:
        xgb_acc = train_xgb(ticker, feature_df)
        all_metrics["xgb_directional_accuracy"] = round(xgb_acc, 2)
    except Exception as exc:
        logger.warning("XGBoost training failed for %s: %s", ticker, exc)

    _save_metrics(all_metrics, ticker)
    logger.info("Training complete for %s. Metrics: %s", ticker, all_metrics)
    return all_metrics


def models_exist(ticker: str) -> bool:
    """Return True if both model checkpoints exist for the given ticker."""
    st_path = _model_path(ticker, "short_term")
    lt_path = _model_path(ticker, "long_term")
    return os.path.exists(st_path) and os.path.exists(lt_path)


def load_trained_models(ticker: str) -> Tuple[StockLSTM, StockLSTM]:
    """
    Load both trained model checkpoints from disk.

    Parameters
    ----------
    ticker : str

    Returns
    -------
    Tuple[StockLSTM, StockLSTM]
        (short_term_model, long_term_model) loaded and ready for inference.

    Raises
    ------
    FileNotFoundError
        If model files are not found.
    """
    st_path = _model_path(ticker, "short_term")
    lt_path = _model_path(ticker, "long_term")

    for path in (st_path, lt_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}. "
                f"Train the model for {ticker} first."
            )

    st_model = build_short_term_model()
    lt_model = build_long_term_model()

    def _load(model: StockLSTM, path: str, label: str) -> None:
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        except RuntimeError as exc:
            if "size mismatch" in str(exc):
                raise FileNotFoundError(
                    f"{ticker} {label} model at {path} was saved with an incompatible "
                    "architecture (output layer size mismatch). Replace the file with a "
                    "seq2seq-trained checkpoint or click 'Retrain Model' in the sidebar."
                ) from exc
            raise RuntimeError(f"Failed to load {label} weights from {path}: {exc}") from exc

    _load(st_model, st_path, "short_term")
    _load(lt_model, lt_path, "long_term")

    st_model.eval()
    lt_model.eval()

    logger.info("Loaded trained models for %s", ticker)
    return st_model, lt_model
