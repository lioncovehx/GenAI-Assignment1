"""Optuna search for the corruption classifier (score: best val cross-entropy).

  python -m task2_hard_routing.tune_classifier --trials 40    # rerun to resume
  python -m task2_hard_routing.tune_classifier --plots
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
from task2_hard_routing.train_classifier import DEFAULT, OUT, train

STORAGE = f"sqlite:///{ROOT / 'optuna' / 'task2_classifier.db'}"
STUDY = "task2_classifier"
EPOCHS = 10  # same budget as the ablations, so trial scores compare with the baseline
SEARCHED = ("lr", "batch_size", "channels", "dropout", "weight_decay")


def suggest(trial: optuna.Trial) -> dict:
    """Optuna search space. Batch sizes are multiples of 4 (balanced batches)."""
    return dict(
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        channels=trial.suggest_categorical("channels", ["16-32-64-128", "32-64-128-256", "32-64-128-128-256"]),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    )


def objective(trial: optuna.Trial) -> float:
    run = train(suggest(trial), EPOCHS, run_name=f"trial_{trial.number:03d}")
    best = float("inf")
    for epoch, score, f1 in run:
        if epoch == 1:  # link the Optuna trial to its MLflow run
            trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)
        if score < best:
            best = score
            trial.set_user_attr("macro_f1", f1)  # at the best-loss epoch: the [Guo17] safeguard
        trial.report(score, epoch)
        if trial.should_prune():
            run.close()  # ends the MLflow run, tagged pruned=true
            raise optuna.TrialPruned()
    return best  # the score of the checkpoint train() keeps


def study() -> optuna.Study:
    (ROOT / "optuna").mkdir(exist_ok=True)
    s = optuna.create_study(
        study_name=STUDY, storage=STORAGE, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        # val scores swung hard in epochs 2-7 of the ablations: judge only from epoch 5,
        # once 5 trials have completed
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
    if not s.trials:  # trial 0 = the ablation baseline, so the search starts from a known-good point
        s.enqueue_trial({k: DEFAULT[k] for k in SEARCHED})
    return s


def plots(s: optuna.Study):
    states = [t.state for t in s.trials]
    print(f"trials: {len(states)} total, {states.count(TrialState.COMPLETE)} complete, "
          f"{states.count(TrialState.PRUNED)} pruned, {states.count(TrialState.FAIL)} failed")
    b = s.best_trial
    print(f"best trial #{b.number}: val loss={b.value:.4f} params={b.params} mlflow_run={b.user_attrs.get('mlflow_run_id')}")
    # [Guo17] safeguard: loss and accuracy can disagree, so show the macro-F1 of the top trials side by side
    done = sorted((t for t in s.trials if t.state == TrialState.COMPLETE), key=lambda t: t.value)[:5]
    for t in done:
        f1 = t.user_attrs.get("macro_f1", float("nan"))
        print(f"  top-5 #{t.number}: val loss={t.value:.4f} macro-F1={f1:.4f} (~{round((1 - f1) * 736)} errors / 736)")
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in (("history", oplt.plot_optimization_history), ("importance", oplt.plot_param_importances),
                     ("slice", oplt.plot_slice)):
        ax = fn(s)
        fig = (ax.flat[0] if hasattr(ax, "flat") else ax).figure
        fig.set_size_inches(16, 4) if name == "slice" else fig.set_size_inches(8, 5)
        fig.tight_layout()
        fig.savefig(fig_dir / f"optuna_classifier_{name}.png", dpi=130)
        print("saved", fig_dir / f"optuna_classifier_{name}.png")


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
        mlflow.set_experiment("task2_routing")
        with mlflow.start_run(run_name="optuna_task2_classifier"):
            mlflow.log_params({"n_trials": a.trials, "epochs_per_trial": EPOCHS, "sampler": "TPE(seed=42)",
                               "pruner": "Median(startup=5, warmup=5)", "trial_0": "ablation baseline (enqueued)"})
            # catch: a failed trial (e.g. the 5-min data-loader timeout) is recorded and the study moves on
            s.optimize(objective, catch=(RuntimeError,), callbacks=[MaxTrialsCallback(a.trials, states=None)])
        plots(s)
