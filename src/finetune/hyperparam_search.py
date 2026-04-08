"""
Hyperparameter search script for ChatterboxTTS fine-tuning.

This script supports multiple search strategies:
1. Grid Search - Test all combinations of specified values
2. Random Search - Randomly sample from parameter ranges
3. Optuna - Bayesian optimization for smarter search

Usage:
    python -m finetune.hyperparam_search --strategy grid
    python -m finetune.hyperparam_search --strategy random --n_trials 20
    python -m finetune.hyperparam_search --strategy optuna --n_trials 50
"""

import argparse
import copy
import itertools
import logging
import os
import random
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ============================================================================
# HYPERPARAMETER SEARCH SPACE CONFIGURATION
# ============================================================================

# Define your hyperparameter search space here
SEARCH_SPACE = {
    # Learning rate - log scale is common for LR
    "learning_rate": {
        "type": "categorical",  # or "loguniform", "uniform"
        "values": [1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
    },
    # Warmup steps
    "warmup_steps": {
        "type": "categorical",
        "values": [100, 200, 500, 1000],
    },
    # Max gradient norm for clipping
    "max_grad_norm": {
        "type": "categorical",
        "values": [0.5, 1.0, 2.0, 5.0],
    },
    "gradient_accumulation_steps": {
        "type": "categorical",
        "values": [2, 4, 6],
    },
}

# For Optuna continuous search (more granular)
OPTUNA_SEARCH_SPACE = {
    "learning_rate": {
        "type": "loguniform",
        "low": 1e-6,
        "high": 5e-4,
    },
    "warmup_steps": {
        "type": "int",
        "low": 100,
        "high": 1000,
        "step": 100,
    },
    "max_grad_norm": {
        "type": "uniform",
        "low": 0.5,
        "high": 5.0,
    },
    "gradient_accumulation_steps": {
        "type": "categorical",
        "values": [2, 4, 6],
    },
}


def load_base_config(config_path: str) -> Dict[str, Any]:
    """Load the base YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to a YAML file."""
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config_with_params(
    base_config: Dict[str, Any], params: Dict[str, Any], run_name: str, output_dir: str
) -> Dict[str, Any]:
    """Update base config with hyperparameters for a specific trial."""
    config = copy.deepcopy(base_config)

    # Update hyperparameters
    for key, value in params.items():
        config[key] = value

    # Update run-specific settings
    config["run_name"] = run_name
    config["output_dir"] = output_dir

    return config


def run_training(config_path: str, dry_run: bool = False) -> Optional[float]:
    """
    Run training with the specified config and return the final eval loss.

    Returns:
        The final evaluation loss, or None if training failed.
    """
    cmd = [
        sys.executable,
        "-m",
        "finetune.finetune_t3",
        "--config",
        config_path,
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    if dry_run:
        logger.info("DRY RUN - Would execute training")
        return random.uniform(0.5, 2.0)  # Fake loss for testing

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,  # Run from src directory
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Training completed successfully")

        # Try to extract eval loss from output or metrics file
        # You may need to adjust this based on your actual output format
        eval_loss = extract_eval_loss(result.stdout)
        return eval_loss

    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with error: {e.stderr}")
        return None


def extract_eval_loss(output: str) -> Optional[float]:
    """
    Extract evaluation loss from training output.
    Adjust this function based on your actual logging format.
    """
    import re

    # Look for patterns like "eval_loss': 0.1234" or "eval_loss: 0.1234"
    patterns = [
        r"'eval_loss':\s*([\d.]+)",
        r"eval_loss:\s*([\d.]+)",
        r"eval_loss\s*=\s*([\d.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))

    logger.warning("Could not extract eval loss from output")
    return None


# ============================================================================
# GRID SEARCH
# ============================================================================


def grid_search(
    base_config: Dict[str, Any],
    search_space: Dict[str, Dict],
    output_base_dir: str,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    Perform grid search over all combinations of hyperparameters.
    """
    # Extract all parameter values
    param_names = list(search_space.keys())
    param_values = [search_space[name]["values"] for name in param_names]

    # Generate all combinations
    combinations = list(itertools.product(*param_values))
    total_trials = len(combinations)

    logger.info(f"Grid search: {total_trials} total combinations")

    results = []
    for trial_idx, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))

        # Create unique run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        param_str = "_".join([f"{k[:3]}{v}" for k, v in params.items()])
        run_name = f"grid_{trial_idx:03d}_{param_str}"

        # Create output directory for this trial
        trial_output_dir = os.path.join(output_base_dir, run_name)
        os.makedirs(trial_output_dir, exist_ok=True)

        # Update config
        config = update_config_with_params(base_config, params, run_name, trial_output_dir)

        # Save trial config
        trial_config_path = os.path.join(trial_output_dir, "config.yaml")
        save_config(config, trial_config_path)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Trial {trial_idx + 1}/{total_trials}")
        logger.info(f"Parameters: {params}")
        logger.info(f"{'=' * 60}")

        # Run training
        eval_loss = run_training(trial_config_path, dry_run=dry_run)

        result = {
            "trial_idx": trial_idx,
            "params": params,
            "eval_loss": eval_loss,
            "config_path": trial_config_path,
            "output_dir": trial_output_dir,
        }
        results.append(result)

        # Save intermediate results
        save_results(results, os.path.join(output_base_dir, "grid_search_results.yaml"))

    return results


# ============================================================================
# RANDOM SEARCH
# ============================================================================


def sample_from_space(search_space: Dict[str, Dict]) -> Dict[str, Any]:
    """Sample a single configuration from the search space."""
    params = {}
    for name, spec in search_space.items():
        if spec["type"] == "categorical":
            params[name] = random.choice(spec["values"])
        elif spec["type"] == "uniform":
            params[name] = random.uniform(spec["low"], spec["high"])
        elif spec["type"] == "loguniform":
            import math

            log_low = math.log(spec["low"])
            log_high = math.log(spec["high"])
            params[name] = math.exp(random.uniform(log_low, log_high))
        elif spec["type"] == "int":
            step = spec.get("step", 1)
            params[name] = random.randrange(spec["low"], spec["high"] + 1, step)
    return params


def random_search(
    base_config: Dict[str, Any],
    search_space: Dict[str, Dict],
    output_base_dir: str,
    n_trials: int = 20,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    Perform random search over the hyperparameter space.
    """
    logger.info(f"Random search: {n_trials} trials")

    results = []
    for trial_idx in range(n_trials):
        params = sample_from_space(search_space)

        # Round float values for cleaner logging
        display_params = {k: round(v, 6) if isinstance(v, float) else v for k, v in params.items()}

        # Create unique run name
        run_name = f"random_{trial_idx:03d}_{datetime.now().strftime('%H%M%S')}"

        # Create output directory for this trial
        trial_output_dir = os.path.join(output_base_dir, run_name)
        os.makedirs(trial_output_dir, exist_ok=True)

        # Update config
        config = update_config_with_params(base_config, params, run_name, trial_output_dir)

        # Save trial config
        trial_config_path = os.path.join(trial_output_dir, "config.yaml")
        save_config(config, trial_config_path)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Trial {trial_idx + 1}/{n_trials}")
        logger.info(f"Parameters: {display_params}")
        logger.info(f"{'=' * 60}")

        # Run training
        eval_loss = run_training(trial_config_path, dry_run=dry_run)

        result = {
            "trial_idx": trial_idx,
            "params": display_params,
            "eval_loss": eval_loss,
            "config_path": trial_config_path,
            "output_dir": trial_output_dir,
        }
        results.append(result)

        # Save intermediate results
        save_results(results, os.path.join(output_base_dir, "random_search_results.yaml"))

    return results


# ============================================================================
# OPTUNA SEARCH (Bayesian Optimization)
# ============================================================================


def optuna_search(
    base_config: Dict[str, Any],
    search_space: Dict[str, Dict],
    output_base_dir: str,
    n_trials: int = 50,
    dry_run: bool = False,
    study_name: str = "chatterbox_hyperparam_search",
) -> List[Dict[str, Any]]:
    """
    Perform Bayesian optimization using Optuna.
    Optuna learns from previous trials to suggest better hyperparameters.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        logger.error("Optuna not installed. Install with: pip install optuna")
        sys.exit(1)

    results = []

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters using Optuna's suggest methods
        params = {}
        for name, spec in search_space.items():
            if spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["values"])
            elif spec["type"] == "uniform":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif spec["type"] == "loguniform":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
            elif spec["type"] == "int":
                step = spec.get("step", 1)
                params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=step)

        # Round float values for cleaner output
        display_params = {k: round(v, 6) if isinstance(v, float) else v for k, v in params.items()}

        # Create unique run name
        run_name = f"optuna_{trial.number:03d}"

        # Create output directory for this trial
        trial_output_dir = os.path.join(output_base_dir, run_name)
        os.makedirs(trial_output_dir, exist_ok=True)

        # Update config
        config = update_config_with_params(base_config, params, run_name, trial_output_dir)

        # Save trial config
        trial_config_path = os.path.join(trial_output_dir, "config.yaml")
        save_config(config, trial_config_path)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Optuna Trial {trial.number + 1}/{n_trials}")
        logger.info(f"Parameters: {display_params}")
        logger.info(f"{'=' * 60}")

        # Run training
        eval_loss = run_training(trial_config_path, dry_run=dry_run)

        if eval_loss is None:
            # Return a high loss for failed trials
            eval_loss = float("inf")

        result = {
            "trial_idx": trial.number,
            "params": display_params,
            "eval_loss": eval_loss,
            "config_path": trial_config_path,
            "output_dir": trial_output_dir,
        }
        results.append(result)

        # Save intermediate results
        save_results(results, os.path.join(output_base_dir, "optuna_search_results.yaml"))

        return eval_loss

    # Create Optuna study
    sampler = TPESampler(seed=42)  # TPE = Tree-structured Parzen Estimator
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # We want to minimize eval loss
        sampler=sampler,
        storage=f"sqlite:///{output_base_dir}/optuna_study.db",  # Persist study
        load_if_exists=True,  # Resume if study exists
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Log best trial
    logger.info(f"\n{'=' * 60}")
    logger.info("BEST TRIAL:")
    logger.info(f"  Value (eval_loss): {study.best_trial.value}")
    logger.info(f"  Params: {study.best_trial.params}")
    logger.info(f"{'=' * 60}")

    return results


# ============================================================================
# RESULTS HANDLING
# ============================================================================


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save search results to a YAML file."""
    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)


def print_results_summary(results: List[Dict[str, Any]]) -> None:
    """Print a summary of the search results."""
    # Filter out failed trials
    valid_results = [r for r in results if r["eval_loss"] is not None and r["eval_loss"] != float("inf")]

    if not valid_results:
        logger.info("No valid results to summarize")
        return

    # Sort by eval loss
    sorted_results = sorted(valid_results, key=lambda x: x["eval_loss"])

    logger.info(f"\n{'=' * 60}")
    logger.info("SEARCH RESULTS SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total trials: {len(results)}")
    logger.info(f"Successful trials: {len(valid_results)}")

    logger.info(f"\nTop 5 configurations:")
    for i, result in enumerate(sorted_results[:5]):
        logger.info(f"\n  Rank {i + 1}:")
        logger.info(f"    Eval Loss: {result['eval_loss']:.6f}")
        logger.info(f"    Params: {result['params']}")
        logger.info(f"    Output: {result['output_dir']}")

    # Best configuration
    best = sorted_results[0]
    logger.info(f"\n{'=' * 60}")
    logger.info("BEST CONFIGURATION:")
    logger.info(f"  Eval Loss: {best['eval_loss']:.6f}")
    for k, v in best["params"].items():
        logger.info(f"  {k}: {v}")
    logger.info(f"{'=' * 60}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for ChatterboxTTS fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="finetune/configs/finetune_test.yaml",
        help="Path to base YAML configuration file",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["grid", "random", "optuna"],
        default="optuna",
        help="Search strategy to use",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of trials for random/optuna search",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/hyperparam_search",
        help="Base directory for search outputs",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't actually run training, just test the search logic",
    )
    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # Load base config
    base_config = load_base_config(str(config_path))

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = os.path.join(args.output_dir, f"{args.strategy}_{timestamp}")
    os.makedirs(output_base_dir, exist_ok=True)

    logger.info(f"Hyperparameter search: {args.strategy}")
    logger.info(f"Base config: {config_path}")
    logger.info(f"Output directory: {output_base_dir}")

    # Save base config for reference
    save_config(base_config, os.path.join(output_base_dir, "base_config.yaml"))

    # Run search
    if args.strategy == "grid":
        results = grid_search(
            base_config,
            SEARCH_SPACE,
            output_base_dir,
            dry_run=args.dry_run,
        )
    elif args.strategy == "random":
        results = random_search(
            base_config,
            SEARCH_SPACE,
            output_base_dir,
            n_trials=args.n_trials,
            dry_run=args.dry_run,
        )
    elif args.strategy == "optuna":
        results = optuna_search(
            base_config,
            OPTUNA_SEARCH_SPACE,
            output_base_dir,
            n_trials=args.n_trials,
            dry_run=args.dry_run,
        )

    # Print summary
    print_results_summary(results)

    logger.info(f"\nAll results saved to: {output_base_dir}")


if __name__ == "__main__":
    main()
