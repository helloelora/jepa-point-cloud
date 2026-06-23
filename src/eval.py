"""PointCloud — downstream evaluation (answers the view-invariance question).

The feature-extraction harness is provided. What you implement (`# TODO`) is the
linear probe + metric on the official ModelNet40 test split, and the comparison
that makes the result meaningful: the frozen SSL encoder vs a random-encoder floor
(and ideally the same probe across rotate=none|z|so3 to expose the invariance gap).

Run:  python -m examples.pointcloud.eval --ckpt <.../latest.pth.tar>
"""
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.pointcloud.dataset import PointCloudConfig, PointCloudDataset
from examples.pointcloud.main import build_encoder


@torch.no_grad()
def extract_features(encoder, split, dcfg, device):
    """Provided: frozen encoder -> [N, D] features + labels for `split`.

    Uses the deterministic clean (supervised-mode) view so the probe sees one
    canonical sampling per shape."""
    cfg = PointCloudConfig(**{**dcfg, "split": split, "mode": "supervised"})
    ds = PointCloudDataset(cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=8)
    X, y = [], []
    for xb, yb in loader:
        X.append(encoder.represent(xb.to(device)).cpu().numpy())
        y.append(np.asarray(yb))
    return np.concatenate(X), np.concatenate(y)


# --------------------------------------------------------------------------- #
# PROBE + METRIC  — # TODO
# --------------------------------------------------------------------------- #
def probe(Xtr, ytr, Xte, yte, n_classes):
    """Linear probe on FROZEN features: standardize on train only (no leakage),
    fit multinomial LogisticRegression, score top-1 accuracy on the official test
    split, and report against chance (100 / n_classes = 2.5% for ModelNet40)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)          # fit on TRAIN only -> no test leakage
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    clf = LogisticRegression(max_iter=1000)     # lbfgs multinomial for multiclass
    clf.fit(Xtr_s, ytr)

    acc = float((clf.predict(Xte_s) == yte).mean())
    chance = 1.0 / n_classes
    return {
        "accuracy": acc,
        "accuracy_pct": round(100.0 * acc, 2),
        "chance": chance,
        "chance_pct": round(100.0 * chance, 2),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_classes": int(n_classes),
        "feat_dim": int(Xtr.shape[1]),
    }


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(state["encoder"]); encoder.eval()

    dcfg = OmegaConf.to_container(cfg.data, resolve=True)
    Xtr, ytr = extract_features(encoder, "train", dcfg, device)
    Xte, yte = extract_features(encoder, "test", dcfg, device)
    print("[pointcloud-eval]", probe(Xtr, ytr, Xte, yte, dcfg["n_classes"]))


if __name__ == "__main__":
    main()
