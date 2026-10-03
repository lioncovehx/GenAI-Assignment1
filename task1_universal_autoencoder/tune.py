"""Optuna search for the universal autoencoder (score: SSIM + PSNR/40 at the best epoch).

  python -m task1_universal_autoencoder.tune --trials 40    # rerun to resume
  python -m task1_universal_autoencoder.tune --plots
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
from task1_universal_autoencoder.train import OUT, train

STORAGE = f"sqlite:///{ROOT / 'optuna' / 'task1.db'}"
STUDY = "task1_universal"
EPOCHS = 15  # same budget as the ablations


def suggest(trial: optuna.Trial) -> dict:
    """Optuna search space."""
    return dict(
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        latent_ch=trial.suggest_categorical("latent_ch", [16, 32, 64]),
        base_ch=trial.suggest_categorical("base_ch", [32, 48, 64]),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        alpha=trial.suggest_float("alpha", 0.5, 0.95),
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
    return optuna.create_study(
        study_name=STUDY, storage=STORAGE, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        # >= 40 trials -> Hyperband; weak trials stop at epoch 3 or 5
        pruner=optuna.pruners.HyperbandPruner(min_resource=3, max_resource=EPOCHS, reduction_factor=3))


def plots(s: optuna.Study):
    states = [t.state for t in s.trials]
    print(f"trials: {len(states)} total, {states.count(TrialState.COMPLETE)} complete, "
          f"{states.count(TrialState.PRUNED)} pruned, {states.count(TrialState.FAIL)} failed")
    b = s.best_trial
    print(f"best trial #{b.number}: objective={b.value:.4f} params={b.params} mlflow_run={b.user_attrs.get('mlflow_run_id')}")
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in (("history", oplt.plot_optimization_history), ("importance", oplt.plot_param_importances),
                     ("slice", oplt.plot_slice)):
        ax = fn(s)
        fig = (ax.flat[0] if hasattr(ax, "flat") else ax).figure
        fig.set_size_inches(16, 4) if name == "slice" else fig.set_size_inches(8, 5)
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
        mlflow.set_experiment("task1_universal")
        with mlflow.start_run(run_name="optuna_task1"):
            mlflow.log_params({"n_trials": a.trials, "epochs_per_trial": EPOCHS, "sampler": "TPE(seed=42)",
                               "pruner": "Hyperband(3, 15, 3)"})
            # catch: a failed trial (e.g. the 5-min data-loader timeout) is recorded and the study moves on
            s.optimize(objective, catch=(RuntimeError,), callbacks=[MaxTrialsCallback(a.trials, states=None)])
        plots(s)
