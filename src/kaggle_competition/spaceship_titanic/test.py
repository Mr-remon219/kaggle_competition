from torch.utils.data import DataLoader
import torch
import pandas as pd

from .data_utils import TestDataset, tta_predict
from .model.resnet1D import ResNet1D
from .config import DATA_DIR, MODEL_DIR, OUTPUT_DIR, set_seed


def test():
    set_seed(42)
    data_dir = DATA_DIR
    model_path = MODEL_DIR / "model.pth"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "submission.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TestDataset()
    test_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model = ResNet1D(1, 2, dropout=0.3)

    check_point = torch.load(model_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in check_point:
        state_dict = check_point["model_state_dict"]
    else:
        state_dict = check_point
    model.load_state_dict(state_dict)
    model = model.to(device)

    preds = []
    for data in test_loader:
        data = data.to(device)
        batch_pred = tta_predict(model, data, num_aug=7).argmax(dim=1)
        preds.extend(batch_pred.cpu().numpy())

    sub_df = pd.read_csv(data_dir / "sample_submission.csv")
    sub_df["Transported"] = pd.Series(preds).astype(bool)
    sub_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    test()
