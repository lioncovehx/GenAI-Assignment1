"""Test-set evaluation of Task 2: classifier metrics, and oracle vs predicted routing against Task 1.

  python -m task2_hard_routing.evaluate
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from common import gpu, metrics
from common.corruptions import LABELS
from common.data import ROOT
from common.datasets import ManifestDataset
from task1_universal_autoencoder.model import ConvAE
from task1_universal_autoencoder.train import MODEL_KEYS, OUT as T1_OUT
from task2_hard_routing.router import EXPERTS, load_classifier, load_specialist
from task2_hard_routing.train_classifier import OUT, save_confusion

TEST_OUT = OUT / "test"
OUTPUTS = ("input", "universal", *EXPERTS)  # everything computed for every row
SEV = ["-", "low", "medium", "high"]


def load_universal(dev) -> ConvAE:
    ck = torch.load(T1_OUT / "final" / "best.pt", map_location="cpu", weights_only=False)
    m = ConvAE(**{k: ck["cfg"][k] for k in MODEL_KEYS})
    m.load_state_dict(ck["state_dict"])
    return m.to(dev).eval()


def pick(df, route_col, metric):
    """Per row: the metric of the output selected by the route (0 = input, k = EXPERTS[k-1])."""
    cols = [f"{metric}_{o}" for o in ("input", *EXPERTS)]
    return df[cols].to_numpy()[np.arange(len(df)), df[route_col].to_numpy()]


@torch.no_grad()
def run(ds, dev):
    mf = torch.channels_last
    clf = load_classifier(dev).to(memory_format=mf)
    nets = {"universal": load_universal(dev), **{k: load_specialist(k, dev) for k in EXPERTS}}
    for n in nets.values():
        n.to(memory_format=mf)
    rows = []
    loader = DataLoader(ds, 256, num_workers=8, multiprocessing_context="forkserver", timeout=300)
    for xc, x, label, idx in loader:
        xc_d, x_d = xc.to(dev, memory_format=mf), x.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = clf(xc_d).float().softmax(1)
            outs = {"input": xc_d, **{k: n(xc_d).float() for k, n in nets.items()}}
        sc = {o: metrics.per_image(out, x_d) for o, out in outs.items()}
        p, pred = p.cpu(), p.argmax(1).cpu()
        for i, j in enumerate(idx.tolist()):
            r = ds.rows[j]
            rows.append({"row": j, "image": r["image"], "name": r["name"], "severity": r["severity"] or "-",
                         "label": int(label[i]), "pred": int(pred[i]),
                         **{f"p_{n}": float(p[i, k]) for k, n in enumerate(LABELS)},
                         **{f"{m}_{o}": float(sc[o][m][i]) for o in OUTPUTS for m in ("psnr", "ssim")}})
    df = pd.DataFrame(rows)
    for m in ("psnr", "ssim"):
        df[f"{m}_oracle"] = pick(df, "label", m)
        df[f"{m}_predicted"] = pick(df, "pred", m)
    return df


def ordered(t):
    return t.reindex(sorted(t.index, key=lambda k: (LABELS.index(k[0]), SEV.index(k[1]))) if isinstance(t.index, pd.MultiIndex)
                     else [n for n in LABELS if n in t.index])


@torch.no_grad()
def misroute_figure(df, ds, dev, path, n=8):
    """The n misrouted rows with the largest SSIM cost (fixed rule; SSIM because an untouched clean image scores the
    100 dB PSNR cap, which would inflate clean->X costs). Columns: clean | input | oracle | predicted."""
    worst = df[df.pred != df.label].nlargest(n, "cost_ssim")
    nets = {k: load_specialist(k, dev) for k in EXPERTS}
    fig, ax = plt.subplots(len(worst), 4, figsize=(8, 2.1 * len(worst)), squeeze=False)
    for r, (_, w) in enumerate(worst.iterrows()):
        xc, x, _, _ = ds[int(w.row)]
        run1 = lambda k: xc if k == 0 else nets[EXPERTS[k - 1]](xc[None].to(dev))[0].float().cpu().clamp(0, 1)
        panels = [(x, "clean"), (xc, f"input {w.psnr_input:.1f} dB"),
                  (run1(int(w.label)), f"oracle: {LABELS[int(w.label)]}\n{w.psnr_oracle:.1f} dB"),
                  (run1(int(w.pred)), f"predicted: {LABELS[int(w.pred)]} (p={w[f'p_{LABELS[int(w.pred)]}']:.2f})\n{w.psnr_predicted:.1f} dB")]
        for c, (im, title) in enumerate(panels):
            ax[r, c].imshow(im.permute(1, 2, 0).numpy())
            ax[r, c].set_title(title, fontsize=7)
            ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
        ax[r, 0].set_ylabel(f"img {int(w.image)}\n{w['name']} {w.severity}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return worst


def main():
    dev = gpu.setup()
    TEST_OUT.mkdir(parents=True, exist_ok=True)
    ds = ManifestDataset("test")
    df = run(ds, dev)
    df["cost_psnr"] = df.psnr_oracle - df.psnr_predicted
    df["cost_ssim"] = df.ssim_oracle - df.ssim_predicted
    df.to_csv(TEST_OUT / "per_image.csv", index=False)
    pd.set_option("display.width", 220)

    # 1. classifier
    rep = pd.DataFrame(classification_report(df.label, df.pred, labels=range(4), target_names=LABELS,
                                             output_dict=True, digits=4)).T.round(4)
    rep.to_csv(TEST_OUT / "classifier_report.csv")
    acc = float((df.label == df.pred).mean())
    save_confusion(df.label, df.pred, TEST_OUT / "confusion_test.png",
                   f"Task 2 classifier: TEST ({len(df):,} images), accuracy {acc:.4f}")
    sev = df.assign(correct=df.label == df.pred).groupby(["name", "severity"]).correct.agg(["size", "mean"])
    sev = ordered(sev.rename(columns={"size": "n", "mean": "recall"})).round(4)
    sev.to_csv(TEST_OUT / "recall_by_severity.csv")
    print(f"== classifier (test) == accuracy {acc:.4f}\n{rep.to_string()}\n\n== recall by severity ==\n{sev.to_string()}")

    # 2. restoration: by type and by type x severity
    methods = ["input", "universal", "oracle", "predicted"]
    cols = [f"{m}_{o}" for m in ("psnr", "ssim") for o in methods]
    by_type = ordered(df.groupby("name")[cols].mean()).round(3)
    by_type["untouched_%"] = ordered(df.groupby("name").pred.apply(lambda s: (s == 0).mean() * 100)).round(2)
    by_sev = ordered(df.groupby(["name", "severity"])[cols].mean()).round(3)
    by_type.to_csv(TEST_OUT / "table_by_type.csv")
    by_sev.to_csv(TEST_OUT / "table_by_severity.csv")
    print(f"\n== restoration by type (PSNR capped at 100 for untouched images) ==\n{by_type.to_string()}")
    print(f"\n== restoration by type x severity ==\n{by_sev.to_string()}")

    # 3. cross-test: every specialist (and the universal model) on every corruption type, mean PSNR / SSIM
    cross = ordered(df.groupby("name")[[f"{m}_{o}" for m in ("psnr", "ssim") for o in OUTPUTS]].mean()).T.round(3)
    cross.to_csv(TEST_OUT / "cross_test.csv")
    print(f"\n== cross-test (rows = model, columns = true corruption) ==\n{cross.to_string()}")

    # 4. misrouting
    mis = df[df.pred != df.label]
    err = (mis.assign(error=mis.name + " -> " + mis.pred.map(dict(enumerate(LABELS))))
           .groupby("error").agg(n=("row", "size"), cost_psnr=("cost_psnr", "mean"), cost_ssim=("cost_ssim", "mean"),
                                 psnr_oracle=("psnr_oracle", "mean"), psnr_predicted=("psnr_predicted", "mean"))
           .sort_values("n", ascending=False).round(3))
    err.to_csv(TEST_OUT / "misrouting.csv")
    print(f"\n== misrouting: {len(mis)} of {len(df)} rows ({len(mis) / len(df):.2%}) ==\n{err.to_string()}")
    worst = misroute_figure(df, ds, dev, TEST_OUT / "misroutes.png")
    print("\n== worst misroutes (largest SSIM cost) ==\n" + worst[["image", "name", "severity", "pred", "psnr_input",
          "psnr_oracle", "psnr_predicted", "cost_psnr", "ssim_oracle", "ssim_predicted", "cost_ssim"]].round(2).to_string(index=False))

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("task2_routing")
    with mlflow.start_run(run_name="test_eval"):
        mlflow.log_params({"test_rows": len(df), "classifier": "outputs/final/best.pt",
                           "specialists": "outputs/final_{salt,blur,occlusion}/best.pt"})
        mlflow.log_metrics({"test_acc": acc, "test_macro_f1": float(rep.loc["macro avg", "f1-score"]),
                            "test_misrouted": len(mis), "test_cost_psnr_mean": float(df.cost_psnr.mean()),
                            **{f"test_{m}_{o}_{n}": float(by_type.loc[n, f"{m}_{o}"]) for n in by_type.index
                               for m in ("psnr", "ssim") for o in methods}})
        mlflow.log_artifacts(str(TEST_OUT), "test")
    print(f"\nsaved {TEST_OUT}")


if __name__ == "__main__":
    main()
