"""Optuna search for the sketch cGAN (score: val SSIM - LPIPS at the best epoch).

  python -m task4_cgan_sketch.tune --trials 40    # rerun to resume
  python -m task4_cgan_sketch.tune --plots
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
from task4_cgan_sketch.train import DEFAULT, OUT, train

STORAGE = f"sqlite:///{ROOT / 'optuna' / 'task4.db'}"
STUDY = "task4_sketch"
EPOCHS = 40  # same budget as the ablations, so trial scores compare with the baseline's 0.342


def suggest(trial: optuna.Trial) -> dict:
    """Optuna search space."""
    return dict(
        lr_G=trial.suggest_float("lr_G", 5e-5, 5e-4, log=True),
        lr_D=trial.suggest_float("lr_D", 5e-5, 5e-4, log=True),
        batch_size=trial.suggest_categorical("batch_size", [1, 4, 8, 16]),
        base_ch=trial.suggest_categorical("base_ch", [32, 48, 64]),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        emb_dim=trial.suggest_categorical("emb_dim", [8, 16, 32, 64]),
        lam_l1=trial.suggest_float("lam_l1", 10.0, 200.0, log=True),
    )


def objective(trial: optuna.Trial) -> float:
    run = train(suggest(trial), EPOCHS, run_name=f"trial_{trial.number:03d}")
    best = -float("inf")
    for epoch, score in run:
        if epoch == 1:  # link the Optuna trial to its MLflow run
            trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)
        best = max(best, score)
        trial.report(score, epoch)
        if trial.should_prune():
            run.close()  # ends the MLflow run, tagged pruned=true
            raise optuna.TrialPruned()
    return best  # the score of the checkpoint train() keeps


def study() -> optuna.Study:
    (ROOT / "optuna").mkdir(exist_ok=True)
    s = optuna.create_study(
        study_name=STUDY, storage=STORAGE, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        # GAN curves are noisy early: only judge from epoch 10, and only after 8 complete trials
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=10))
    if not s.trials:  # trial 0 = the ablation baseline, so the search starts from a known-good point
        s.enqueue_trial({k: DEFAULT[k] for k in ("lr_G", "lr_D", "batch_size", "base_ch", "dropout", "emb_dim", "lam_l1")})
    return s


def plots(s: optuna.Study):
    states = [t.state for t in s.trials]
    print(f"trials: {len(states)} total, {states.count(TrialState.COMPLETE)} complete, "
          f"{states.count(TrialState.PRUNED)} pruned, {states.count(TrialState.FAIL)} failed")
    b = s.best_trial
    print(f"best trial #{b.number}: objective={b.value:.4f} params={b.params} mlflow_run={b.user_attrs.get('mlflow_run_id')}")
    for base in (32, 48, 64):  # best per model size: the ONNX file size trade-off
        done = [t for t in s.trials if t.state == TrialState.COMPLETE and t.params["base_ch"] == base]
        if done:
            t = max(done, key=lambda t: t.value)
            print(f"  best with base_ch={base}: trial #{t.number}, objective={t.value:.4f}")
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
        mlflow.set_experiment("task4_sketch")
        with mlflow.start_run(run_name="optuna_task4"):
            mlflow.log_params({"n_trials": a.trials, "epochs_per_trial": EPOCHS, "sampler": "TPE(seed=42)",
                               "pruner": "Median(startup=8, warmup=10)", "trial_0": "ablation baseline (enqueued)"})
            # catch: a failed trial (e.g. the 5-min data-loader timeout) is recorded and the study moves on
            s.optimize(objective, catch=(RuntimeError,), callbacks=[MaxTrialsCallback(a.trials, states=None)])
        plots(s)
