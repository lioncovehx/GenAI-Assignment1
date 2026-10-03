"""Optuna search for the soft MoE. Collapsed runs (a branch weight above 0.7 or below 0.03) are pruned.

  python -m task3_soft_moe.tune --trials 40    # rerun to resume
  python -m task3_soft_moe.tune --plots
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import mlflow
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.visualization import matplotlib as oplt

from common.data import ROOT
from task3_soft_moe.train import BRANCHES, DEFAULT, OUT, train

STORAGE = f"sqlite:///{ROOT / 'optuna' / 'task3.db'}"
STUDY = "task3_moe"
EPOCHS = 8  # 2 warm-up + 6 joint, the baseline's budget (~1.6 min)
SEARCHED = ("lr_joint", "tau", "lam_c", "lam_b", "alpha", "lr_gate")


def suggest(trial: optuna.Trial) -> dict:
    """Optuna search space; lam_c down to 1e-3 and lr_gate added after the baseline."""
    return dict(
        lr_joint=trial.suggest_float("lr_joint", 1e-6, 1e-4, log=True),
        tau=trial.suggest_float("tau", 0.3, 3.0, log=True),
        lam_c=trial.suggest_float("lam_c", 1e-3, 1.0, log=True),
        lam_b=trial.suggest_float("lam_b", 1e-4, 1e-1, log=True),
        alpha=trial.suggest_float("alpha", 0.3, 0.95),
        lr_gate=trial.suggest_float("lr_gate", 1e-5, 1e-3, log=True),
    )


def objective(trial: optuna.Trial) -> float:
    run = train(suggest(trial), EPOCHS, run_name=f"trial_{trial.number:03d}")
    best = -float("inf")
    for epoch, score, info in run:
        if epoch == 1:  # link the Optuna trial to its MLflow run
            trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)
        if score > best:  # remember what the best epoch looked like: does it blend?
            best = score
            for k, v in ({k: v for k, v in info.items() if k != "wbar"} | {"best_epoch": epoch}).items():
                trial.set_user_attr(k, v)
        wbar = info["wbar"]
        if wbar.max() > 0.7 or wbar.min() < 0.03:
            trial.set_user_attr("pruned_reason", f"collapse at epoch {epoch}: wbar {dict(zip(BRANCHES, wbar.round(3)))}")
            run.close()
            raise optuna.TrialPruned()
        trial.report(score, epoch)
        if trial.should_prune():
            trial.set_user_attr("pruned_reason", f"median at epoch {epoch}")
            run.close()
            raise optuna.TrialPruned()
    return best


def study() -> optuna.Study:
    (ROOT / "optuna").mkdir(exist_ok=True)
    s = optuna.create_study(
        study_name=STUDY, storage=STORAGE, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=DEFAULT["warmup"] + 1))
    if not s.trials:  # trial 0 = the Step-1 baseline
        s.enqueue_trial({k: DEFAULT[k] for k in SEARCHED})
    return s


def plots(s: optuna.Study):
    states = [t.state for t in s.trials]
    reasons = [t.user_attrs.get("pruned_reason", "") for t in s.trials if t.state == TrialState.PRUNED]
    print(f"trials: {len(states)} total, {states.count(TrialState.COMPLETE)} complete, "
          f"{states.count(TrialState.PRUNED)} pruned ({sum(r.startswith('collapse') for r in reasons)} for collapse), "
          f"{states.count(TrialState.FAIL)} failed")
    for t in s.trials:
        if t.user_attrs.get("pruned_reason", "").startswith("collapse"):
            print(f"  #{t.number} {t.user_attrs['pruned_reason']} params={ {k: round(v, 5) for k, v in t.params.items()} }")
    done = sorted((t for t in s.trials if t.state == TrialState.COMPLETE), key=lambda t: -t.value)[:5]
    for t in done:
        a = t.user_attrs
        print(f"  top-5 #{t.number}: objective={t.value:.4f} (epoch {a.get('best_epoch')}) blur->identity w={a.get('w_blur_identity', 0):.3f} "
              f"psnr salt/blur/occl={a.get('psnr_salt', 0):.2f}/{a.get('psnr_blur', 0):.2f}/{a.get('psnr_occlusion', 0):.2f} "
              f"params={ {k: round(v, 6) for k, v in t.params.items()} }")
    if s.trials[0].state == TrialState.COMPLETE:
        a = s.trials[0].user_attrs
        print(f"  trial 0 (baseline): objective={s.trials[0].value:.4f} blur->identity w={a.get('w_blur_identity', 0):.3f} "
              f"psnr salt/blur/occl={a.get('psnr_salt', 0):.2f}/{a.get('psnr_blur', 0):.2f}/{a.get('psnr_occlusion', 0):.2f}")
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in (("history", oplt.plot_optimization_history), ("importance", oplt.plot_param_importances),
                     ("slice", oplt.plot_slice)):
        ax = fn(s)
        fig = (ax.flat[0] if hasattr(ax, "flat") else ax).figure
        fig.set_size_inches(18, 4) if name == "slice" else fig.set_size_inches(8, 5)
        fig.tight_layout()
        fig.savefig(fig_dir / f"optuna_{name}.png", dpi=130)
        print("saved", fig_dir / f"optuna_{name}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=40, help="total trials in the study (counts earlier ones on resume)")
    p.add_argument("--plots", action="store_true", help="only print the summary and save the report figures")
    a = p.parse_args()
    s = study()
    if a.plots:
        plots(s)
    else:
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
        mlflow.set_experiment("task3_moe")
        with mlflow.start_run(run_name="optuna_task3"):
            mlflow.log_params({"n_trials": a.trials, "epochs_per_trial": EPOCHS, "sampler": "TPE(seed=42)",
                               "pruner": "collapse (wbar >0.7 or <0.03) + Median(startup=5, warmup=3)",
                               "trial_0": "Step-1 baseline (enqueued)"})
            # catch: a failed trial (e.g. the 5-min data-loader timeout) is recorded and the study moves on
            s.optimize(objective, catch=(RuntimeError,), callbacks=[MaxTrialsCallback(a.trials, states=None)])
        plots(s)
