"""
LSTM model definitions for short-term and long-term stock price prediction.
"""
import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
NUM_FEATURES = 20           # Must match len(FEATURE_COLUMNS) in feature_engineering.py

# Short-term model architecture
ST_LOOKBACK = 60            # 60 trading days (~3 months)
ST_HIDDEN_SIZE = 128
ST_NUM_LAYERS = 2
ST_DROPOUT = 0.2
ST_PREDICTION_HORIZON = 30  # Max iterative forward steps

# Long-term model architecture
LT_LOOKBACK = 252           # 252 trading days (~1 year)
LT_HIDDEN_SIZE = 256
LT_NUM_LAYERS = 3
LT_DROPOUT = 0.3
LT_PREDICTION_HORIZON = 252


class StockLSTM(nn.Module):
    """
    Multi-layer stacked LSTM with a feedforward prediction head.

    Architecture:
        Input → [LSTM layer 1] → Dropout → [LSTM layer 2…N] → last hidden state
             → Linear(hidden → mid) → ReLU → [optional Dropout] → Linear(mid → horizon)

    The model predicts the entire forecast horizon in a single forward pass
    (sequence-to-sequence / direct multi-output), which avoids the mean-reversion
    artefact produced by auto-regressive re-feeding.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        head_hidden: int,
        horizon: int,
        head_dropout: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        input_size : int
            Number of features per timestep (must match num columns in feature matrix).
        hidden_size : int
            Number of LSTM units per layer.
        num_layers : int
            Number of stacked LSTM layers.
        dropout : float
            Dropout probability applied between LSTM layers (only active when num_layers > 1).
        head_hidden : int
            Hidden size of the first linear layer in the prediction head.
        horizon : int
            Number of future trading days to predict in one shot.
            Short-term: 30, Long-term: 252.
        head_dropout : float
            Dropout probability inside the prediction head (0.0 = no dropout).
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        # Stacked LSTM: dropout between layers is handled by PyTorch's built-in dropout arg
        # Note: PyTorch's nn.LSTM dropout applies between layers, not after the final layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,           # Input shape: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Feedforward prediction head — final layer outputs `horizon` values at once
        layers = [
            nn.Linear(hidden_size, head_hidden),
            nn.ReLU(),
        ]
        if head_dropout > 0.0:
            layers.append(nn.Dropout(head_dropout))
        layers.append(nn.Linear(head_hidden, horizon))

        self.head = nn.Sequential(*layers)

        # Xavier uniform initialisation for all linear layers
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialisation to all linear layers in the head."""
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — predicts the full horizon in one shot.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, seq_len, input_size).

        Returns
        -------
        torch.Tensor
            Shape (batch_size, horizon) — predicted normalised close prices
            for each of the next `horizon` trading days.
        """
        # h0 and c0 default to zeros when not provided
        lstm_out, _ = self.lstm(x)

        # Use only the output at the last timestep as context for prediction
        last_hidden = lstm_out[:, -1, :]      # shape: (batch_size, hidden_size)

        prediction = self.head(last_hidden)   # shape: (batch_size, horizon)
        return prediction


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_short_term_model() -> StockLSTM:
    """
    Instantiate and return the short-term LSTM model.

    Short-term model characteristics:
    - 2 LSTM layers, 128 hidden units, 0.2 dropout
    - Head: Linear(128→64) → ReLU → Linear(64→30)
    - Predicts all 30 future days in one shot (seq2seq, no iterative re-feeding)

    Returns
    -------
    StockLSTM
        Uninitialised (randomly weighted) short-term model.
    """
    model = StockLSTM(
        input_size=NUM_FEATURES,
        hidden_size=ST_HIDDEN_SIZE,
        num_layers=ST_NUM_LAYERS,
        dropout=ST_DROPOUT,
        head_hidden=64,
        horizon=ST_PREDICTION_HORIZON,
        head_dropout=0.0,
    )
    logger.info(
        "Built short-term model: input=%d, hidden=%d, layers=%d, horizon=%d",
        NUM_FEATURES, ST_HIDDEN_SIZE, ST_NUM_LAYERS, ST_PREDICTION_HORIZON
    )
    return model


def build_long_term_model() -> StockLSTM:
    """
    Instantiate and return the long-term LSTM model.

    Long-term model characteristics:
    - 3 LSTM layers, 256 hidden units, 0.3 dropout
    - Head: Linear(256→128) → ReLU → Dropout(0.2) → Linear(128→252)
    - Predicts all 252 future days in one shot (seq2seq, no iterative re-feeding)

    Returns
    -------
    StockLSTM
        Uninitialised (randomly weighted) long-term model.
    """
    model = StockLSTM(
        input_size=NUM_FEATURES,
        hidden_size=LT_HIDDEN_SIZE,
        num_layers=LT_NUM_LAYERS,
        dropout=LT_DROPOUT,
        head_hidden=128,
        horizon=LT_PREDICTION_HORIZON,
        head_dropout=0.2,
    )
    logger.info(
        "Built long-term model: input=%d, hidden=%d, layers=%d, horizon=%d",
        NUM_FEATURES, LT_HIDDEN_SIZE, LT_NUM_LAYERS, LT_PREDICTION_HORIZON
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
