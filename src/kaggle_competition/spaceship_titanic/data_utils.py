import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from .config import DATA_CONFIG_DIR, DATA_DIR, MODEL_DIR

filepath = {"train": DATA_DIR / "train.csv", "test": DATA_DIR / "test.csv"}

# ---- Column definitions ----
# All feature columns (order matters — this determines the tensor layout).
# Raw spending columns are intentionally excluded — their log1p versions
# capture relative magnitude without the extreme-outlier problem.
FEATURE_COLS = [
    "HomePlanet", "CryoSleep", "Destination", "Age", "VIP",
    "CabinDeck", "CabinRoom", "CabinSide",
    "MemberNum", "GroupSize",
    "RoomService_log", "FoodCourt_log", "ShoppingMall_log", "Spa_log", "VRDeck_log",
    "TotalSpend_log", "HasSpending",
]

# Every feature column gets MinMax-scaled to [0, 1].
# Binary columns (VIP, CryoSleep, HasSpending) are already {0,1} — no-op.
SCALE_COLS = list(FEATURE_COLS)

# Numeric columns eligible for noise injection.
# Binary and low-cardinality categoricals excluded (noise on them is meaningless).
NUMERIC_NOISE_NAMES = [
    "Age",
    "CabinRoom", "MemberNum", "GroupSize",
    "RoomService_log", "FoodCourt_log", "ShoppingMall_log", "Spa_log", "VRDeck_log",
    "TotalSpend_log",
]

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


# =============================================================================
# Feature engineering helpers
# =============================================================================


def _extract_cabin(df: pd.DataFrame) -> None:
    """Split 'B/0/P' → CabinDeck, CabinRoom, CabinSide (raw strings, not factorized yet)."""
    cabin_split = df["Cabin"].fillna("Unknown/0/Unknown").str.split("/", expand=True)
    df["CabinDeck"] = cabin_split[0]
    df["CabinRoom"] = pd.to_numeric(cabin_split[1], errors="coerce").fillna(0).astype(float)
    df["CabinSide"] = cabin_split[2]
    df.drop(columns=["Cabin"], inplace=True)


def _extract_group_features(df: pd.DataFrame) -> None:
    """Split '0001_01' → GroupId (raw string), MemberNum, GroupSize."""
    split = df["PassengerId"].str.split("_", expand=True)
    df["GroupId"] = split[0]
    df["MemberNum"] = pd.to_numeric(split[1], errors="coerce").fillna(1).astype(int)
    group_sizes = df.groupby("GroupId").size()
    df["GroupSize"] = df["GroupId"].map(group_sizes).astype(int)
    df.drop(columns=["PassengerId", "Name"], inplace=True)


def _engineer_spending(df: pd.DataFrame) -> None:
    """Create log-transformed and aggregated spending features."""
    for c in SPEND_COLS:
        df[f"{c}_log"] = np.log1p(df[c].clip(lower=0))
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)
    df["TotalSpend_log"] = np.log1p(df["TotalSpend"].clip(lower=0))
    df["HasSpending"] = (df["TotalSpend"] > 0).astype(int)


# =============================================================================
# Scaling
# =============================================================================


def _minmax_scale(
    df: pd.DataFrame, columns: list[str], scaler_params: dict | None = None
) -> tuple[pd.DataFrame, dict] | pd.DataFrame:
    """MinMaxScaler: (x - min) / (max - min) → [0, 1].

    Returns (df, scaler_params) in fit mode, or df in transform mode.
    """
    if scaler_params is None:
        # ---- Fit mode ----
        scaler_params = {}
        for col in columns:
            vmin, vmax = float(df[col].min()), float(df[col].max())
            if vmax - vmin < 1e-10:
                # Constant column — avoid division by zero
                df[col] = 0.0
                scaler_params[col] = {"min": vmin, "max": vmin + 1.0}
            else:
                scaler_params[col] = {"min": vmin, "max": vmax}
                df[col] = (df[col] - vmin) / (vmax - vmin)
        return df, scaler_params
    else:
        # ---- Transform mode ----
        for col in columns:
            p = scaler_params[col]
            df[col] = ((df[col] - p["min"]) / (p["max"] - p["min"])).clip(0, 1)
        return df


# =============================================================================
# Main data pipeline
# =============================================================================


def data_init(file_path):
    """Load raw CSV, run all feature engineering, impute, factorize, scale.

    On train data: fits and saves a MinMaxScaler to MODEL_DIR/scaler.pkl.
    On test data:  loads the cached scaler and applies it.
    """
    df = pd.read_csv(file_path)

    # ---- 1. Fix VRDeck bug: ensure numeric ----
    df["VRDeck"] = pd.to_numeric(df["VRDeck"], errors="coerce")

    # ---- 2. Feature engineering ----
    _extract_cabin(df)
    _extract_group_features(df)  # also drops Name (redundant with GroupId)

    # ---- 3. Age=0 → missing ----
    df.loc[df["Age"] == 0, "Age"] = np.nan

    # ---- 4. Smart NaN filling (before factorize) ----
    # CryoSleep: fill NaN with mode (False)
    df["CryoSleep"] = df["CryoSleep"].fillna(False)

    # Spending: CryoSleep passengers → 0, others → median
    for c in SPEND_COLS:
        cryo_mask = df["CryoSleep"] == True  # noqa: E712 (must use == for pandas bool)
        df.loc[cryo_mask & df[c].isna(), c] = 0.0
        # median of non-cryo non-null values, fallback 0
        non_cryo_median = df.loc[~cryo_mask, c].median()
        if pd.isna(non_cryo_median):
            non_cryo_median = 0.0
        df[c] = df[c].fillna(non_cryo_median)

    # Age: fill by HomePlanet median, then global median fallback
    df["Age"] = df["Age"].fillna(
        df.groupby("HomePlanet")["Age"].transform("median")
    )
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Other categoricals: fill NaN with mode
    for c in ["HomePlanet", "Destination", "VIP"]:
        modes = df[c].mode()
        if len(modes) > 0:
            df[c] = df[c].fillna(modes[0])

    # ---- 5. Spending feature engineering (after NaN are filled) ----
    _engineer_spending(df)

    # ---- 6. Factorize categoricals (fit on train → save; load → apply on test) ----
    factorize_cols = [
        "HomePlanet", "CryoSleep", "Destination",
        "CabinDeck", "CabinSide",
    ]
    factorize_path = DATA_CONFIG_DIR / "factorize.pkl"

    if file_path == filepath["train"]:
        DATA_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        factorize_mappings = {}
        for c in factorize_cols:
            df[c], uniques = pd.factorize(df[c])
            factorize_mappings[c] = list(uniques)
        with open(factorize_path, "wb") as f:
            pickle.dump(factorize_mappings, f)
    else:
        with open(factorize_path, "rb") as f:
            factorize_mappings = pickle.load(f)
        for c in factorize_cols:
            codes = pd.Categorical(df[c], categories=factorize_mappings[c]).codes
            df[c] = codes  # unseen categories become -1

    # ---- 7. Bool → int ----
    bool_cols = ["VIP"]
    if "Transported" in df.columns:
        bool_cols.append("Transported")
    for c in bool_cols:
        df[c] = df[c].replace({False: 0, True: 1}).astype(int)

    # ---- 8. Fill any remaining NaN (safety net for engineered features) ----
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype in (np.float64, np.float32, np.int64, np.int32):
                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
            else:
                modes = df[col].mode()
                df[col] = df[col].fillna(modes[0] if len(modes) > 0 else 0)

    # ---- 9. Scale numeric columns ----
    scaler_path = DATA_CONFIG_DIR / "scaler.pkl"

    if file_path == filepath["train"]:
        df, scaler_params = _minmax_scale(df, SCALE_COLS, scaler_params=None)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler_params, f)
    else:
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}. Run train.py first to fit the scaler."
            )
        with open(scaler_path, "rb") as f:
            scaler_params = pickle.load(f)
        df = _minmax_scale(df, SCALE_COLS, scaler_params=scaler_params)

    return df


# =============================================================================
# Group-aware split
# =============================================================================


def _group_shuffle_split(groups: np.ndarray, test_size: float = 0.15, random_state: int = 42):
    """Split indices so the same group stays entirely in train or val."""
    rng = np.random.RandomState(random_state)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    n_test = max(1, int(len(unique_groups) * test_size))
    test_groups = set(unique_groups[:n_test])
    train_idx = [i for i, g in enumerate(groups) if g not in test_groups]
    val_idx = [i for i, g in enumerate(groups) if g in test_groups]
    return train_idx, val_idx


# =============================================================================
# TTA (Test-Time Augmentation)
# =============================================================================


def tta_predict(model: torch.nn.Module, x: torch.Tensor,
                num_aug: int = 7, noise_scale: float = 0.03) -> torch.Tensor:
    """N 次噪声增强 → 平均 softmax 概率（eval 模式，无 MC Dropout）。

    Returns:
        Tensor of shape [B, num_classes] — averaged softmax probabilities.
    """
    indices = [FEATURE_COLS.index(n) for n in NUMERIC_NOISE_NAMES]
    std = x[:, indices].std(dim=0).clamp(min=1e-8)

    probs = []
    with torch.no_grad():
        for _ in range(num_aug):
            x_aug = x.clone()
            noise = torch.randn(x.shape[0], len(indices), device=x.device) * std * noise_scale
            x_aug[:, indices] += noise
            x_aug[:, indices] = x_aug[:, indices].clamp(0, 1)
            probs.append(F.softmax(model(x_aug.unsqueeze(1)), dim=1))

    return torch.stack(probs).mean(dim=0)


# =============================================================================
# PyTorch Datasets
# =============================================================================


class TrainDataset(Dataset):
    """Training dataset with optional Gaussian noise augmentation."""

    def __init__(self, df: pd.DataFrame, training: bool = True):
        super().__init__()
        self.training = training
        self.df = df.reset_index(drop=True)
        self.feature = self.df[FEATURE_COLS].to_numpy(dtype=np.float32)
        self.label = self.df["Transported"].to_numpy(dtype=np.int64)

        # Precompute noise std for numeric columns (used in __getitem__)
        self.numeric_indices = [FEATURE_COLS.index(n) for n in NUMERIC_NOISE_NAMES]
        numeric_vals = self.feature[:, self.numeric_indices]
        self.numeric_std = torch.tensor(
            np.nanstd(numeric_vals, axis=0), dtype=torch.float32
        )

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        x = torch.tensor(self.feature[idx], dtype=torch.float32)
        y = torch.tensor(self.label[idx], dtype=torch.long)

        if self.training:
            # Gaussian noise: ~2% of per-column std, clipped to [0, 1]
            noise = torch.randn(len(self.numeric_indices)) * self.numeric_std * 0.02
            x[self.numeric_indices] += noise
            x[self.numeric_indices] = x[self.numeric_indices].clamp(0, 1)

        return x, y


class TestDataset(Dataset):
    """Test dataset — no labels, no noise."""

    def __init__(self):
        super().__init__()
        df = data_init(filepath["test"])
        self.feature = df[FEATURE_COLS].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        x = torch.tensor(self.feature[idx], dtype=torch.float32)
        return x


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    train_df = data_init(filepath["train"])
    print(f"Train shape: {train_df.shape}")
    print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Label distribution:\n{train_df['Transported'].value_counts()}")

    train_dataset = TrainDataset(train_df, training=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    for x, y in train_loader:
        print(f"Batch x: {x.shape}, y: {y.shape}, y unique: {y.unique()}")
        break

    test_dataset = TestDataset()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    for x in test_loader:
        print(f"Test batch x: {x.shape}")
        break
