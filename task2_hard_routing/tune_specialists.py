"""Optuna search shared by the three specialists. Each trial trains salt, blur, then occlusion;
the score is the mean of their best val scores.

  python -m task2_hard_routing.tune_specialists --trials 30    # rerun to resume
  python -m task2_hard_routing.tune_specialists --plots
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
from task2_hard_routing.train_specialist import OUT, SPECIALIST, TYPES, train

STORAGE = f"sqlite:///{ROOT / 'optuna' / 'task2_specialists.db'}"
STUDY = "task2_specialists"
EPOCHS = 10  # per specialist, so 30 epochs per trial
SEARCHED = ("lr", "batch_size", "latent_ch", "base_ch", "alpha")


def suggest(trial: optuna.Trial) -> dict:
    """Task 1's ranges, except alpha starts at 0.3: Task 1's best alpha (0.54) sat near its 0.5 limit."""
    return dict(
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        latent_ch=trial.suggest_categorical("latent_ch", [16, 32, 64]),
        base_ch=trial.suggest_categorical("base_ch", [32, 48, 64]),
        alpha=trial.suggest_float("alpha", 0.3, 0.95),
    )


def objective(trial: optuna.Trial) -> float:
    cfg = suggest(trial)
    bests = []
    for k, kind in enumerate(TYPES):
        run = train(kind, cfg, EPOCHS, run_name=f"trial_{trial.number:03d}_{kind}")
        best = -float("inf")
        for epoch, score in run:
            if epoch == 1:  # link the Optuna trial to its MLflow runs
                trial.set_user_attr(f"mlflow_run_{kind}", mlflow.active_run().info.run_id)
            best = max(best, score)
        trial.set_user_attr(f"best_{kind}", best)
        bests.append(best)
        if k < len(TYPES) - 1:  # judge after salt (step 10) and after blur (step 20); occlusion = the final value
            trial.report(best, 10 * (k + 1))
            if trial.should_prune():
                raise optuna.TrialPruned()
    return sum(bests) / len(bests)


def study() -> optuna.Study:
    (ROOT / "optuna").mkdir(exist_ok=True)
    s = optuna.create_study(
        study_name=STUDY, storage=STORAGE, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    if not s.trials:  # trial 0 = Task 1's recipe: does the universal model's best setting suit specialists?
        s.enqueue_trial({k: SPECIALIST[k] for k in SEARCHED})
    return s


def plots(s: optuna.Study):
    states = [t.state for t in s.trials]
    print(f"trials: {len(states)} total, {states.count(TrialState.COMPLETE)} complete, "
          f"{states.count(TrialState.PRUNED)} pruned, {states.count(TrialState.FAIL)} failed")
    done = sorted((t for t in s.trials if t.state == TrialState.COMPLETE), key=lambda t: -t.value)[:5]
    for t in done:
        per = " ".join(f"{k}={t.user_attrs.get(f'best_{k}', float('nan')):.4f}" for k in TYPES)
        print(f"  top-5 #{t.number}: mean={t.value:.4f} ({per}) params={t.params}")
    t0 = s.trials[0]
    if t0.state == TrialState.COMPLETE:
        print(f"  trial 0 (Task 1 settings): mean={t0.value:.4f}")
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in (("history", oplt.plot_optimization_history), ("importance", oplt.plot_param_importances),
                     ("slice", oplt.plot_slice)):
        ax = fn(s)
        fig = (ax.flat[0] if hasattr(ax, "flat") else ax).figure
        fig.set_size_inches(16, 4) if name == "slice" else fig.set_size_inches(8, 5)
        fig.tight_layout()
        fig.savefig(fig_dir / f"optuna_specialists_{name}.png", dpi=130)
        print("saved", fig_dir / f"optuna_specialists_{name}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30, help="total trials in the study (counts earlier ones on resume)")
    p.add_argument("--plots", action="store_true", help="only print the summary and save the report figures")
    a = p.parse_args()
    s = study()
    if a.plots:
        plots(s)
    else:
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
        mlflow.set_experiment("task2_routing")
        with mlflow.start_run(run_name="optuna_task2_specialists"):
            mlflow.log_params({"n_trials": a.trials, "epochs_per_specialist": EPOCHS, "sampler": "TPE(seed=42)",
                               "pruner": "Median(startup=5), judged after each whole specialist (steps 10, 20)",
                               "trial_0": "Task 1 final settings (enqueued)"})
            # catch: a failed trial (e.g. the 5-min data-loader timeout) is recorded and the study moves on
            s.optimize(objective, catch=(RuntimeError,), callbacks=[MaxTrialsCallback(a.trials, states=None)])
        plots(s)
