import argparse
import sys
from pathlib import Path

# Enable `from common.logging_config import ...` before anything else imports logging.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/
from common.logging_config import setup_logging  # noqa: E402

setup_logging()

from train.train import run as train_run  # noqa: E402
from validation.validation import run as eval_run  # noqa: E402
from validation.validation_tflite import run as eval_tflite_run  # noqa: E402
from test.test import run as test_run  # noqa: E402
import tools.hardware_test as ht  # noqa: E402
from tools.configuration_generator import config  # noqa: E402
from tools.config_validator import validate_config  # noqa: E402

ACTIONS = {
    "train": train_run,
    "eval": eval_run,
    "eval_tflite": eval_tflite_run,
    "test": test_run,
}

def init(action: str, model_name: str):
    ht.check_requirement()
    ht.check_python_version()
    ht.check_gpus()

    config(model_name=model_name)

    cfg = validate_config()

    ht.check_train_dataset_dir(cfg)
    ht.check_train_results_dir(cfg)

    func = ACTIONS[action]
    func(cfg)

def parse_args():
    parser = argparse.ArgumentParser(description="MobileNetV3-Small pipeline runner")

    parser.add_argument(
        "--action", "--actions",
        dest="action",
        required=True,
        choices=ACTIONS.keys(),
        help="What to do: train | eval | eval_tflite | test",
    )

    parser.add_argument(
        "--model",
        default="mobilenet-v3-small",
        help="Model name to use, e.g. mobilenet-v3-small",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    init(args.action, args.model)
