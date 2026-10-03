"""Test-set evaluation of the soft MoE against Tasks 1 and 2, plus gate weights, collapse and mixed corruptions.

  python -m task3_soft_moe.evaluate [--run final]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import corruptions as C
from common import gpu, metrics
from common.corruptions import LABELS
from common.data import ROOT
from common.datasets import CleanDataset, ManifestDataset
from task2_hard_routing.evaluate import load_universal
from task2_hard_routing.router import EXPERTS, load_classifier, load_specialist
from task2_hard_routing.train_classifier import save_confusion
from task3_soft_moe.moe import build
from task3_soft_moe.train import BRANCHES, OUT, save_heatmap

METHODS = ("input", "task1", "t2_oracle", "t2_pred", "t3")
SEV = ["-", "low", "medium", "high"]
MIXES = {  # two corruptions on one image, fixed levels, applied in this order
    "salt+blur": [(2, {"k": 5, "sigma": 1.5}), (1, {"p": 0.05})],
    "blur+occlusion": [(2, {"k": 5, "sigma": 1.5}), (3, {"n": 1, "coverage": 0.20})],
}


def load_moe(run: str, dev):
    ck = torch.load(OUT / run / "best.pt", map_location="cpu", weights_only=False)
    moe = build(ck["cfg"]["tau"], "cpu")
    moe.load_state_dict(ck["state_dict"])
    print(f"Task 3 model: outputs/{run}/best.pt (epoch {ck['epoch']}, val objective {ck['objective']:.4f}, tau {ck['cfg']['tau']:.3f})")
    return moe.to(dev).eval(), ck


class Models:
    """Every network once, on the GPU, in eval mode."""

    def __init__(self, run, dev):
        self.dev = dev
        self.task1 = load_universal(dev)
        self.clf = load_classifier(dev)
        self.t2 = [load_specialist(k, dev) for k in EXPERTS]
        self.moe, self.ck = load_moe(run, dev)

    @torch.no_grad()
    def run(self, xc):
        """-> dict of outputs [B,3,H,W] per method + t3 weights, t3 expert outputs, t2 route, t3 gate argmax."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t2_outs = [xc.float()] + [e(xc).float() for e in self.t2]  # branch 0 = identity
            route = self.clf(xc).float().argmax(1)
            logits = self.moe.gate(xc).float()
            w = (logits / self.moe.tau).softmax(1)
            t3_outs = [xc.float()] + [e(xc).float() for e in self.moe.experts]
            task1 = self.task1(xc).float()
        t3 = (w[:, :, None, None, None] * torch.stack(t3_outs, 1)).sum(1)
        stack2 = torch.stack(t2_outs, 1)
        return {"input": xc.float(), "task1": task1, "t2_stack": stack2, "t2_route": route,
                "t3": t3, "w": w, "t3_gate": logits.argmax(1), "t3_outs": t3_outs}


@torch.no_grad()
def test_rows(models, ds):
    rows = []
    loader = DataLoader(ds, 128, num_workers=8, multiprocessing_context="forkserver", timeout=300)
    for xc, x, label, idx in loader:
        xc, x, label = xc.to(models.dev), x.to(models.dev), label.to(models.dev)
        o = models.run(xc)
        ar = torch.arange(len(label), device=label.device)
        outs = {"input": o["input"], "task1": o["task1"], "t3": o["t3"],
                "t2_oracle": o["t2_stack"][ar, label], "t2_pred": o["t2_stack"][ar, o["t2_route"]]}
        sc = {m: metrics.per_image(v, x) for m, v in outs.items()}
        drift = {b: metrics.per_image(v, x)["psnr"] for b, v in zip(BRANCHES, o["t3_outs"]) if b != "identity"}
        for i, j in enumerate(idx.tolist()):
            r = ds.rows[j]
            rows.append({"row": j, "image": r["image"], "name": r["name"], "severity": r["severity"] or "-",
                         "label": int(label[i]), "t2_route": int(o["t2_route"][i]), "t3_gate": int(o["t3_gate"][i]),
                         **{f"w_{b}": float(o["w"][i, k]) for k, b in enumerate(BRANCHES)},
                         **{f"{m}_{o_}": float(sc[o_][m][i]) for o_ in METHODS for m in ("psnr", "ssim")},
                         **{f"expert_{b}_psnr": float(v[i]) for b, v in drift.items()}})
    return pd.DataFrame(rows)


def ordered(t):
    if isinstance(t.index, pd.MultiIndex):
        return t.reindex(sorted(t.index, key=lambda k: (LABELS.index(k[0]), SEV.index(k[1]))))
    return t.reindex([n for n in LABELS if n in t.index])


@torch.no_grad()
def examples_figure(df, ds, models, path):
    """Fixed rule: 4 'dominant' rows (max weight > 0.9) and 4 'distributed' rows (max weight < 0.6), one per
    true class where possible, each the first such row in manifest order."""
    wcols = [f"w_{b}" for b in BRANCHES]
    df = df.assign(wmax=df[wcols].max(axis=1))
    picks = []
    for kind, mask in (("dominant", df.wmax > 0.9), ("distributed", df.wmax < 0.6)):
        sub = df[mask]
        chosen = [sub[sub.name == n].iloc[0] for n in LABELS if (sub.name == n).any()][:4]
        if len(chosen) < 4:  # fill from any class
            chosen += [r for _, r in sub[~sub.row.isin([c.row for c in chosen])].head(4 - len(chosen)).iterrows()]
        picks += [(kind, r) for r in chosen]
    fig, ax = plt.subplots(len(picks), 4, figsize=(10, 2.2 * len(picks)), squeeze=False,
                           gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    for r, (kind, w) in enumerate(picks):
        xc, x, _, _ = ds[int(w.row)]
        out = models.run(xc[None].to(models.dev))["t3"][0].cpu().clamp(0, 1)
        for c, (im, title) in enumerate([(xc, f"input ({w['name']} {w.severity})"), (out, f"Task 3 {w.psnr_t3:.1f} dB"), (x, "clean")]):
            ax[r, c].imshow(im.permute(1, 2, 0).numpy())
            ax[r, c].set_title(title, fontsize=7)
            ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
        vals = [w[f"w_{b}"] for b in BRANCHES]
        ax[r, 3].barh(list(BRANCHES)[::-1], vals[::-1], color=["#5E97C4" if v == max(vals) else "#2A2C2F" for v in vals[::-1]])
        ax[r, 3].set_xlim(0, 1)
        ax[r, 3].set_title(f"{kind}: weights", fontsize=7)
        ax[r, 3].tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def weight_distributions(df, path):
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for k, n in enumerate(LABELS):
        sub = df[df.name == n]
        ax[k].boxplot([sub[f"w_{b}"] for b in BRANCHES], tick_labels=list(BRANCHES), showfliers=True,
                      flierprops={"markersize": 2})
        ax[k].set_title(f"true: {n} (n={len(sub)})", fontsize=9)
        ax[k].tick_params(labelsize=8)
    ax[0].set_ylabel("weight")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def mixed_test(models, n=300, seed=42):
    """D-3.6 on VAL images (the test set has only single corruptions)."""
    clean = CleanDataset("val")
    idx = np.random.default_rng(seed).choice(len(clean), size=min(n, len(clean)), replace=False)
    rows = []
    for mix, steps in MIXES.items():
        rng = np.random.default_rng(seed)
        x = torch.stack([clean[int(i)] for i in idx])
        xc = x.clone()
        for label, level in steps:
            xc = torch.stack([C.apply(img, C.make_params(label, rng, level)) for img in xc])
        o = models.run(xc.to(models.dev))
        x = x.to(models.dev)
        outs = {"input": o["input"], "task1": o["task1"], "t3": o["t3"],
                "t2_pred": o["t2_stack"][torch.arange(len(x), device=x.device), o["t2_route"]]}
        row = {"mix": mix, "n": len(x)}
        for m, v in outs.items():
            s = metrics.per_image(v, x)
            row[f"{m}_psnr"], row[f"{m}_ssim"] = float(s["psnr"].mean()), float(s["ssim"].mean())
        row |= {f"t3_w_{b}": float(o["w"][:, k].mean()) for k, b in enumerate(BRANCHES)}
        row |= {f"t2_route_{n}": float((o["t2_route"] == k).float().mean()) for k, n in enumerate(LABELS)}
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="final")
    a = p.parse_args()
    dev = gpu.setup()
    test_out = OUT / "test"
    test_out.mkdir(parents=True, exist_ok=True)
    models = Models(a.run, dev)
    ds = ManifestDataset("test")
    df = test_rows(models, ds)
    df.to_csv(test_out / "per_image.csv", index=False)
    pd.set_option("display.width", 240)

    # 1. restoration, Task 1 vs Task 2 vs Task 3
    cols = [f"{m}_{o}" for m in ("psnr", "ssim") for o in METHODS]
    by_type = ordered(df.groupby("name")[cols].mean()).round(3)
    by_sev = ordered(df.groupby(["name", "severity"])[cols].mean()).round(3)
    by_type.to_csv(test_out / "table_by_type.csv")
    by_sev.to_csv(test_out / "table_by_severity.csv")
    print(f"== restoration by type (PSNR capped at 100 for untouched images) ==\n{by_type.to_string()}")
    print(f"\n== restoration by type x severity ==\n{by_sev.to_string()}")

    # 2. gate: accuracy, confusion, weights per (type, severity)
    gate_acc = float((df.t3_gate == df.label).mean())
    save_confusion(df.label, df.t3_gate, test_out / "gate_confusion.png", f"Task 3 gate argmax: TEST, accuracy {gate_acc:.4f}")
    wcols = [f"w_{b}" for b in BRANCHES]
    heat = ordered(df.groupby(["name", "severity"])[wcols].mean())
    heat.columns = list(BRANCHES)
    heat.round(4).to_csv(test_out / "gating_heatmap.csv")
    heat.index = [f"{n} {s}" if s != "-" else n for n, s in heat.index]
    save_heatmap(heat, test_out / "gating_heatmap.png", "Task 3: mean branch weight per corruption and severity (test)")
    weight_distributions(df, test_out / "weight_distributions.png")
    print(f"\n== gate accuracy (argmax = true class) {gate_acc:.4f} ==\n== mean weights ==\n{heat.round(3).to_string()}")

    # 3. collapse audit: is any branch dead, or dominating inputs that are not its own?
    audit = []
    for k, b in enumerate(BRANCHES):
        own = df.label == k
        audit.append({"branch": b, "mean_w_own_class": df.loc[own, f"w_{b}"].mean(),
                      "mean_w_other_classes": df.loc[~own, f"w_{b}"].mean(), "max_w_anywhere": df[f"w_{b}"].max(),
                      "share_argmax": float((df[wcols].to_numpy().argmax(1) == k).mean()),
                      "share_other_class_w_over_0.5": float((df.loc[~own, f"w_{b}"] > 0.5).mean())})
    audit = pd.DataFrame(audit).round(4)
    audit.to_csv(test_out / "collapse_audit.csv", index=False)
    print(f"\n== collapse audit ==\n{audit.to_string(index=False)}")

    # 4. specialisation drift: fine-tuned experts on every corruption, next to Task 2's original cross-test
    drift = ordered(df.groupby("name")[[f"expert_{b}_psnr" for b in BRANCHES if b != "identity"]].mean()).T.round(3)
    t2_cross = ROOT / "task2_hard_routing/outputs/test/cross_test.csv"
    if t2_cross.exists():
        t2 = pd.read_csv(t2_cross, index_col=0).loc[[f"psnr_{b}" for b in EXPERTS]]
        t2.index = [f"task2_{b}" for b in EXPERTS]
        drift = pd.concat([drift, t2[drift.columns]])
    drift.to_csv(test_out / "specialisation_drift.csv")
    print(f"\n== specialisation drift (mean PSNR; rows = expert, columns = true corruption) ==\n{drift.to_string()}")

    # 5. examples + mixed corruptions
    examples_figure(df, ds, models, test_out / "examples_dominant_distributed.png")
    mixed = mixed_test(models)
    mixed.to_csv(test_out / "mixed_corruptions.csv", index=False)
    print(f"\n== mixed corruptions (val images) ==\n{mixed.T.to_string()}")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("task3_moe")
    with mlflow.start_run(run_name="test_eval"):
        mlflow.log_params({"checkpoint": f"outputs/{a.run}/best.pt", "checkpoint_epoch": models.ck["epoch"],
                           "test_rows": len(df), "mixed_val_images": int(mixed.n.iloc[0])})
        mlflow.log_metrics({"test_gate_acc": gate_acc,
                            **{f"test_{m}_{o}_{n}": float(by_type.loc[n, f"{m}_{o}"]) for n in by_type.index
                               for m in ("psnr", "ssim") for o in METHODS},
                            **{f"mixed_{r.mix.replace('+', '_')}_{m}_psnr": float(r[f"{m}_psnr"])
                               for _, r in mixed.iterrows() for m in ("input", "task1", "t2_pred", "t3")}})
        mlflow.log_artifacts(str(test_out), "test")
    print(f"\nsaved {test_out}")


if __name__ == "__main__":
    main()
