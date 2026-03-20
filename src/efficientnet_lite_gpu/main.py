from train.train import run as train_run
from validation.validation import run as eval_run
from test.test import run as test_run
import tools.hardware_test as ht
from tools.configuration_generator import config
import sys
import argparse
from tools.config_validator import validate_config

ACTIONS = {
    "train": train_run,
    "eval": eval_run,
    "test": test_run,
}

def init(action: str, model_name: str):
    # Point d'entrée unique : prépare la config puis délègue à l'action demandée.

    # Check environment
    # ht.check_requirement()
    # ht.check_python_version()
    # ht.check_tf_gpu_usable()

    # Create configuration file
    config(model_name=model_name)

    # check valid config
    cfg = validate_config()

    # check train dataset dir
    # ht.check_train_dataset_dir(cfg)

    # check train results dir
    # ht.check_train_results_dir(cfg)

    # Résolution de l'action CLI vers la fonction métier correspondante.
    func = ACTIONS.get(action)

    if func is None:
        print(f"Unknown action: {action}")
        print("Available actions: train | eval | test")
        sys.exit(1)

    # On transmet la même configuration validée à tous les sous-modules.
    func(cfg)

def parse_args():
    parser = argparse.ArgumentParser(description="EfficientNet pipeline runner")

    parser.add_argument(
        "--action", "--actions",
        dest="action",
        required=True,
        choices=ACTIONS.keys(),
        help="What to do: train | eval | test",
    )

    parser.add_argument(
        "--model",
        default="efficientnet-b0",
        help="Model name to use, e.g. efficientnet-b0",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    init(args.action, args.model)
