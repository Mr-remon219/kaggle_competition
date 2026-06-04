"""Ensemble: RF + SVM + XGBoost + CatBoost → soft voting.

ResNet loaded for comparison only — not part of the vote (0.7998 drags it down).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from .data_utils import data_init, FEATURE_COLS, filepath, _group_shuffle_split
from .model.resnet1D import ResNet1D
from .config import MODEL_DIR, OUTPUT_DIR, set_seed


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load & split ----
    print("Loading data...")
    train_df = data_init(filepath["train"])
    groups = train_df["GroupId"].values
    train_idx, val_idx = _group_shuffle_split(groups, test_size=0.15, random_state=42)

    X_train = train_df[FEATURE_COLS].iloc[train_idx].to_numpy()
    y_train = train_df["Transported"].iloc[train_idx].to_numpy()
    X_val   = train_df[FEATURE_COLS].iloc[val_idx].to_numpy()
    y_val   = train_df["Transported"].iloc[val_idx].to_numpy()

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Features: {len(FEATURE_COLS)}")

    # =====================================================================
    # 0. ResNet (reference only — NOT in vote)
    # =====================================================================
    model_path = MODEL_DIR / "model.pth"
    if model_path.exists():
        print("\n--- ResNet (reference) ---")
        resnet = ResNet1D(1, 2, dropout=0.3).to(device)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        resnet.load_state_dict(state)
        resnet.eval()

        ds = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=256)
        rn_probs = []
        with torch.no_grad():
            for (x_batch,) in loader:
                rn_probs.append(torch.softmax(
                    resnet(x_batch.to(device).unsqueeze(1)), dim=1).cpu().numpy())
        rn_probs = np.concatenate(rn_probs)
        rn_acc = (rn_probs.argmax(axis=1) == y_val).mean()
        print(f"ResNet Val Accuracy: {rn_acc:.4f}")
    else:
        rn_acc = None
        print("\n--- ResNet: model.pth not found, skipping ---")

    # =====================================================================
    # 1. Random Forest
    # =====================================================================
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=14, min_samples_leaf=4,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_val, y_val)
    print(f"RF  Val Accuracy: {rf_acc:.4f}")

    # =====================================================================
    # 2. SVM
    # =====================================================================
    print("\n--- SVM ---")
    svm = SVC(probability=True, kernel="rbf", class_weight="balanced",
              random_state=42, C=1.0, gamma="scale")
    svm.fit(X_train, y_train)
    svm_acc = svm.score(X_val, y_val)
    print(f"SVM Val Accuracy: {svm_acc:.4f}")

    # =====================================================================
    # 3. XGBoost
    # =====================================================================
    print("\n--- XGBoost ---")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_acc = xgb.score(X_val, y_val)
    print(f"XGB Val Accuracy: {xgb_acc:.4f}")

    # =====================================================================
    # 4. CatBoost
    # =====================================================================
    print("\n--- CatBoost ---")
    cb = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        random_seed=42, verbose=0, allow_writing_files=False,
    )
    cb.fit(X_train, y_train)
    cb_acc = cb.score(X_val, y_val)
    print(f"CB  Val Accuracy: {cb_acc:.4f}")

    # =====================================================================
    # 5. Soft Voting
    # =====================================================================
    print("\n--- Voting Ensemble ---")
    voting = VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("xgb", xgb), ("cb", cb)],
        voting="soft",
    )
    voting.fit(X_train, y_train)
    vote_acc = voting.score(X_val, y_val)
    print(f"Vote Val Accuracy: {vote_acc:.4f}")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 45)
    if rn_acc is not None:
        print(f"  ResNet: {rn_acc:.4f} (reference)")
    print(f"  RF:     {rf_acc:.4f}")
    print(f"  SVM:    {svm_acc:.4f}")
    print(f"  XGB:    {xgb_acc:.4f}")
    print(f"  CB:     {cb_acc:.4f}")
    print(f"  VOTE:   {vote_acc:.4f}")
    print("=" * 45)

    # =====================================================================
    # Predict & save
    # =====================================================================
    print("\nPredicting test set...")
    test_df = data_init(filepath["test"])
    X_test = test_df[FEATURE_COLS].to_numpy()
    preds = voting.predict(X_test).astype(bool)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = OUTPUT_DIR / "submission.csv"
    sub_df = pd.read_csv(filepath["test"].parent / "sample_submission.csv")
    sub_df["Transported"] = preds
    sub_df.to_csv(sub_path, index=False)
    print(f"Saved to {sub_path}")


if __name__ == "__main__":
    main()
