"""
edupamnet_final.py
==================================================================
Clean reference implementation of the EduPAMNet model and its
training procedure, as used in the FINAL evaluation protocol
(Chapters 4-6).

This file is the version reproduced in the appendix. It contains
ONLY the components that are part of the final protocol:

    - ResidualBlock
    - ImprovedSharedEncoder   (shared encoder; retains an INTERNAL
                               self-attention layer -- this is NOT
                               the cross-task attention that was
                               removed)
    - BalancedDecoder         (platform-specific decoder)
    - OptimizedPAMNet         (EduPAMNet: shared encoder + two
                               platform-specific decoders)
    - FocalLoss               (the classification objective)
    - train_pure_classification (focal-loss-only training loop)

Components explored during methodological development (Section 3.4)
and subsequently REMOVED are deliberately omitted here:

    - optimal-transport feature alignment
    - domain-adversarial training (domain discriminator)
    - cross-task attention
    - the multi-objective (OT + adversarial + balance) loss

Reproducibility note
--------------------------------------------------------------
The empirical results reported in the thesis were produced by the
full development implementation, in which the removed modules were
present but dormant (their outputs did not enter the focal-loss
objective). Because instantiating fewer modules changes the order
in which parameters are randomly initialised, re-running this
trimmed version will NOT reproduce the reported numbers bit-for-bit.
This file is provided as a clean, faithful representation of the
final architecture and training objective, not as a re-run script.

Software: PyTorch 1.12.1 (CUDA 11.6), Python 3.9.12 (see Appendix A).
==================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


# ------------------------------------------------------------------
# Residual block (stable gradient flow on sparse educational features)
# ------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Residual block with GELU activations and batch normalisation."""

    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()

        self.main_path = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

        # Projection on the shortcut when dimensions differ
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate * 0.5)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.main_path(x)
        return self.dropout(self.activation(out + residual))


# ------------------------------------------------------------------
# Shared encoder
# ------------------------------------------------------------------
class ImprovedSharedEncoder(nn.Module):
    """
    Shared encoder mapping the 5-D aligned features to a representation
    common to both platforms.

    Note: the self-attention layer below operates WITHIN the encoder
    (over the single feature vector). It is distinct from the
    cross-task attention module that was removed (Section 3.4).
    """

    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(ImprovedSharedEncoder, self).__init__()

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        # Residual block stack (dropout decays across depth)
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.residual_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i + 1],
                              dropout_rate * (0.8 ** i))
            )

        # Internal self-attention (NOT cross-task attention)
        self.self_attention = nn.MultiheadAttention(
            hidden_dims[-1], num_heads=4,
            dropout=dropout_rate * 0.5, batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_dims[-1])
        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        x = self.input_projection(x)

        for block in self.residual_blocks:
            x = block(x)

        x_seq = x.unsqueeze(1)                      # [batch, 1, features]
        attn_out, _ = self.self_attention(x_seq, x_seq, x_seq)
        x_attended = attn_out.squeeze(1)            # [batch, features]

        x = self.layer_norm(x + x_attended)         # residual + layer norm
        return x


# ------------------------------------------------------------------
# Platform-specific balanced decoder
# ------------------------------------------------------------------
class BalancedDecoder(nn.Module):
    """
    Platform-specific decoder with a learnable balance branch that
    corrects the systematic prediction bias arising from class
    imbalance, which differs between platforms.
    """

    def __init__(self, input_dim, platform_name, target_balance=0.4):
        super(BalancedDecoder, self).__init__()
        self.platform_name = platform_name
        self.target_balance = target_balance

        # Main decoding path
        self.main_decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(input_dim // 2, input_dim // 4),
            nn.BatchNorm1d(input_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Balance branch (outputs [-1, 1] for a balance adjustment)
        self.balance_branch = nn.Sequential(
            nn.Linear(input_dim // 4, input_dim // 8),
            nn.GELU(),
            nn.Linear(input_dim // 8, 1),
            nn.Tanh(),
        )

        # Main prediction branch
        self.prediction_branch = nn.Sequential(
            nn.Linear(input_dim // 4, input_dim // 8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 8, 1),
        )

        # Learnable balance weight
        self.balance_weight = nn.Parameter(torch.tensor(0.3))

        # Platform-specific bias adjustment
        if platform_name == "neurips":
            self.bias_adjustment = nn.Parameter(torch.tensor(-0.2))
        else:
            self.bias_adjustment = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        main_features = self.main_decoder(x)

        prediction = self.prediction_branch(main_features)
        balance_adjustment = self.balance_branch(main_features)

        balanced_prediction = (
            prediction
            + torch.sigmoid(self.balance_weight) * balance_adjustment
            + self.bias_adjustment
        )
        return balanced_prediction


# ------------------------------------------------------------------
# EduPAMNet
# ------------------------------------------------------------------
class OptimizedPAMNet(nn.Module):
    """
    EduPAMNet: a shared encoder feeding two platform-specific
    balanced decoders. Trained with a focal-loss classification
    objective only (Section 3.6).
    """

    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(OptimizedPAMNet, self).__init__()

        # Shared encoder
        self.shared_encoder = ImprovedSharedEncoder(input_dim, hidden_dims, dropout_rate)
        feature_dim = self.shared_encoder.output_dim

        # Platform-specific balanced decoders
        self.neurips_decoder = BalancedDecoder(feature_dim, "neurips", target_balance=0.4)
        self.assistments_decoder = BalancedDecoder(feature_dim, "assistments", target_balance=0.4)

        # Weight initialisation
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x, platform="neurips"):
        # Shared encoding
        shared_features = self.shared_encoder(x)

        # Platform-specific decoding
        if platform == "neurips":
            output = self.neurips_decoder(shared_features)
        elif platform == "assistments":
            output = self.assistments_decoder(shared_features)
        else:
            raise ValueError(f"Unknown platform: {platform}")

        # 'shared_features' is returned for the representation-transfer
        # (PTS) analysis; the training loop uses only 'output'.
        return {
            "output": output.squeeze(),
            "shared_features": shared_features,
        }


# Thesis name for the model
EduPAMNet = OptimizedPAMNet


# ------------------------------------------------------------------
# Classification objective
# ------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal loss for class imbalance: down-weights easy, well-classified
    examples and concentrates learning on the harder minority cases.
    """

    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


# ------------------------------------------------------------------
# Training loop (focal-loss only; no domain-adversarial component)
# ------------------------------------------------------------------
def train_pure_classification(model, train_loader, val_loader,
                              platform, num_epochs=50, device="cpu"):
    """
    Pure classification training: optimises the focal loss only,
    with no domain-adversarial or alignment term.

    Hyperparameters (Appendix B):
        optimiser      AdamW, lr=3e-4, weight_decay=1e-4
        scheduler      CosineAnnealingLR, eta_min=1e-5
        gradient clip  max_norm=1.0
        early stopping patience=15 (on validation F1)
        decision threshold 0.5
    """
    model.to(device)

    # Class weight for the positive class (clamped to a moderate range)
    all_y = np.concatenate([y.numpy() for _, y in train_loader])
    pos = float((all_y == 1).sum())
    neg = float((all_y == 0).sum())
    pw = torch.tensor([neg / pos], dtype=torch.float32).to(device)
    pw = torch.clamp(pw, 0.5, 2.0)

    criterion = FocalLoss(gamma=2.0, pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    best_f1 = -1
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b, platform=platform)["output"].squeeze()
            loss = criterion(out, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # ---- validate ----
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X_v, y_v in val_loader:
                out = model(X_v.to(device), platform=platform)["output"]
                prob = torch.sigmoid(out).cpu().numpy().flatten()
                preds.extend((prob > 0.5).astype(int).tolist())
                trues.extend(y_v.numpy().astype(int).tolist())

        val_f1 = f1_score(trues, preds, zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return model
