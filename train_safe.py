import argparse
import os
from datetime import datetime
import random
import git
from matplotlib.pyplot import get

import torch

from slac.algo import LatentPolicySafetyCriticSlac, SafetyCriticSlacAlgorithm
from slac.env import make_safety
from slac.trainer import Trainer
import json
from configuration import get_default_config

def get_git_short_hash():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    sha_first_7_chars = sha[:7]
    return sha_first_7_chars

def main(args):
    config = get_default_config()
    config["domain_name"] = args.domain_name
    config["task_name"] = args.task_name
    config["seed"] = args.seed
    config["num_steps"] = args.num_steps

    env = make_safety(f'{args.domain_name}{"-" if len(args.domain_name) > 0 else ""}{args.task_name}-v0', image_size=config["image_size"], use_pixels=True, action_repeat=config["action_repeat"])
    env_test = make_safety(f'{args.domain_name}{"-" if len(args.domain_name) > 0 else ""}{args.task_name}-v0', image_size=config["image_size"], use_pixels=True, action_repeat=config["action_repeat"])
    short_hash = get_git_short_hash()
    log_dir = os.path.join(
        "logs",
        f"{short_hash}",
        f"{config['domain_name']}-{config['task_name']}",
        f'slac-seed{config["seed"]}-{datetime.now().strftime("%Y%m%d-%H%M")}',
    )

    algo = LatentPolicySafetyCriticSlac(
        num_sequences=config["num_sequences"],
        gamma_c=config["gamma_c"],
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_repeat=config["action_repeat"],
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=config["seed"],
        buffer_size=config["buffer_size"],
        feature_dim=config["feature_dim"],
        z2_dim=config["z2_dim"],
        hidden_units=config["hidden_units"],
        batch_size_latent=config["batch_size_latent"],
        batch_size_sac=config["batch_size_sac"],
        lr_sac=config["lr_sac"],
        lr_latent=config["lr_latent"],
        start_alpha=config["start_alpha"],
        start_lagrange=config["start_lagrange"],
        grad_clip_norm=config["grad_clip_norm"],
        tau=config["tau"],
        image_noise=config["image_noise"],
    )
    trainer = Trainer(
        num_sequences=config["num_sequences"],
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        seed=config["seed"],
        num_steps=config["num_steps"],
        initial_learning_steps=config["initial_learning_steps"],
        initial_collection_steps=config["initial_collection_steps"],
        collect_with_policy=config["collect_with_policy"],
        eval_interval=config["eval_interval"],
        num_eval_episodes=config["num_eval_episodes"],
        action_repeat=config["action_repeat"],
        train_steps_per_iter=config["train_steps_per_iter"],
        env_steps_per_train_step=config["env_steps_per_train_step"]
    )
    trainer.writer.add_text("config", json.dumps(config), 0)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=2 * 10 ** 6, help="Number of training steps")
    parser.add_argument("--domain_name", type=str, default="Safexp", help="Name of the domain")
    parser.add_argument("--task_name", type=str, default="PointGoal1", help="Name of the task")
    parser.add_argument("--seed", type=int, default=314, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Train using GPU with CUDA")
    args = parser.parse_args()
    main(args)

