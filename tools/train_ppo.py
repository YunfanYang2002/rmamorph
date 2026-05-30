import argparse
import json
import os
import sys
from datetime import datetime

import torch

from metamorph.config import cfg
from metamorph.config import dump_cfg
from metamorph.utils import distributed as du
from metamorph.utils import sample as su


def set_cfg_options():
    calculate_max_iters()
    maybe_infer_walkers()
    calculate_max_limbs_joints()


def get_run_mode_tag():
    mode = str(getattr(cfg.MODEL, "CONTEXT_MODE", "none")).strip().lower()
    base = "baseline" if mode == "none" else mode
    if "HistoryContextWrapper" in set(getattr(cfg.MODEL, "WRAPPERS", [])):
        return "{}_history".format(base)
    return "{}_nohistory".format(base)


def get_run_name(timestamp):
    return "run_{}_{}".format(get_run_mode_tag(), timestamp)


def prepare_run_artifacts():
    if not du.is_main_process():
        return

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    run_name = get_run_name(timestamp)
    log_dir = os.path.join(cfg.OUT_DIR, "tensorboard", run_name)
    cfg_dir = os.path.join(log_dir, "cfg")

    print(
        "[TensorBoard] preparing log_dir={} mode_tag={}".format(
            log_dir, get_run_mode_tag()
        ),
        flush=True,
    )

    os.makedirs(cfg_dir, exist_ok=True)
    cfg.RUN_TIMESTAMP = timestamp
    cfg.RUN_NAME = run_name
    cfg.TB_LOG_DIR = log_dir
    cfg.TB_CFG_DIR = cfg_dir

    run_meta = {
        "mode_tag": get_run_mode_tag(),
        "run_name": run_name,
        "timestamp": timestamp,
        "base_lr": cfg.PPO.BASE_LR,
        "gamma": cfg.PPO.GAMMA,
        "encoder_type": getattr(cfg.MODEL, "ENCODER_TYPE", "default"),
        "actor_critic": cfg.MODEL.ACTOR_CRITIC,
        "context_mode": getattr(cfg.MODEL, "CONTEXT_MODE", "none"),
        "wrappers": list(getattr(cfg.MODEL, "WRAPPERS", [])),
        "desc": cfg.DESC,
        "rng_seed": cfg.RNG_SEED,
    }
    with open(os.path.join(cfg_dir, "run_meta_{}.json".format(timestamp)), "w") as f:
        json.dump(run_meta, f, indent=2)
    dump_cfg(os.path.join("tensorboard", run_name, "cfg", "config.yaml"))


def finalize_run_artifacts():
    if not du.is_main_process():
        return

    if not hasattr(cfg, "RUN_NAME"):
        return

    cfg_dir = os.path.join(cfg.OUT_DIR, "tensorboard", cfg.RUN_NAME, "cfg")
    run_meta_path = os.path.join(cfg_dir, "run_meta_{}.json".format(cfg.RUN_TIMESTAMP))
    run_meta = {
        "mode_tag": get_run_mode_tag(),
        "run_name": cfg.RUN_NAME,
        "timestamp": cfg.RUN_TIMESTAMP,
        "base_lr": cfg.PPO.BASE_LR,
        "gamma": cfg.PPO.GAMMA,
        "encoder_type": getattr(cfg.MODEL, "ENCODER_TYPE", "default"),
        "actor_critic": cfg.MODEL.ACTOR_CRITIC,
        "context_mode": getattr(cfg.MODEL, "CONTEXT_MODE", "none"),
        "wrappers": list(getattr(cfg.MODEL, "WRAPPERS", [])),
        "desc": cfg.DESC,
        "rng_seed": cfg.RNG_SEED,
        "max_joints": getattr(cfg.MODEL, "MAX_JOINTS", None),
        "max_limbs": getattr(cfg.MODEL, "MAX_LIMBS", None),
        "walkers": list(cfg.ENV.WALKERS),
    }
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    dump_cfg(os.path.join("tensorboard", cfg.RUN_NAME, "cfg", "config.yaml"))


def calculate_max_limbs_joints():
    if cfg.ENV_NAME != "Unimal-v0":
        return

    from metamorph.utils import file as fu

    num_joints, num_limbs = [], []

    metadata_paths = []
    for agent in cfg.ENV.WALKERS:
        metadata_paths.append(os.path.join(
            cfg.ENV.WALKER_DIR, "metadata", "{}.json".format(agent)
        ))

    for metadata_path in metadata_paths:
        metadata = fu.load_json(metadata_path)
        num_joints.append(metadata["dof"])
        num_limbs.append(metadata["num_limbs"] + 1)

    # Add extra 1 for max_joints; needed for adding edge padding
    cfg.MODEL.MAX_JOINTS = max(num_joints) + 1
    cfg.MODEL.MAX_LIMBS = max(num_limbs) + 1


def calculate_max_iters():
    # Iter here refers to 1 cycle of experience collection and policy update.
    total_num_envs = cfg.PPO.NUM_ENVS * cfg.WORLD_SIZE
    cfg.PPO.MAX_ITERS = (
        int(cfg.PPO.MAX_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // total_num_envs
    )
    cfg.PPO.EARLY_EXIT_MAX_ITERS = (
        int(cfg.PPO.EARLY_EXIT_STATE_ACTION_PAIRS)
        // cfg.PPO.TIMESTEPS
        // total_num_envs
    )


def maybe_infer_walkers():
    if cfg.ENV_NAME != "Unimal-v0":
        return

    # Only infer the walkers if this option was not specified
    if len(cfg.ENV.WALKERS):
        return

    cfg.ENV.WALKERS = [
        xml_file.split(".")[0]
        for xml_file in os.listdir(os.path.join(cfg.ENV.WALKER_DIR, "xml"))
    ]


def get_hparams():
    from metamorph.utils import file as fu
    from metamorph.utils import sweep as swu

    hparam_path = os.path.join(cfg.OUT_DIR, "hparam.json")
    # For local sweep return
    if not os.path.exists(hparam_path):
        return {}

    hparams = {}
    varying_args = fu.load_json(hparam_path)
    flatten_cfg = swu.flatten(cfg)

    for k in varying_args:
        hparams[k] = flatten_cfg[k]

    return hparams


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument(
        "opts",
        help="See morphology/core/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def ppo_train():
    from metamorph.algos.ppo.ppo import PPO

    su.set_seed(cfg.RNG_SEED, idx=cfg.RANK)
    # Configure the CUDNN backend
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    torch.set_num_threads(1)
    PPOTrainer = PPO()
    PPOTrainer.train()
    hparams = get_hparams()
    PPOTrainer.save_rewards(hparams=hparams)
    PPOTrainer.save_model()


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    du.init_distributed_mode()
    prepare_run_artifacts()
    # Set cfg options which are inferred
    set_cfg_options()
    finalize_run_artifacts()
    if du.is_main_process():
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        dump_cfg()
    du.synchronize()

    try:
        ppo_train()
    finally:
        du.destroy_process_group()


if __name__ == "__main__":
    main()
