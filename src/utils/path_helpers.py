from definitions import *


def get_dataset_path(subset, name, split):
    return DATASETS_DIR_PATH.joinpath(f"{subset}/{name}/{split}").as_posix()


def get_metric_dir_path(model, subset, name, split):
    return METRICS_DIR_PATH.joinpath(f"{subset}/{name}/{split}/{model}")
