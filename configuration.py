def get_default_config():
    cfg = {
            "domain_name": "Safexp",
            "task_name": "PointGoal1",
            "action_repeat": 2,
            "image_size": 64,
            "image_noise": 0.4,
            "seed": 0,
            "num_sequences": 10,
            "gamma_c": 0.995,
            "buffer_size": 2*10**5,
            "feature_dim": 200,
            "z2_dim": 200,
            "hidden_units": (256, 256),
            "batch_size_latent": 32,
            "batch_size_sac": 64,
            "lr_sac": 2e-4,
            "lr_latent": 1e-4,
            "start_alpha": 4e-3,
            "start_lagrange": 0.02,
            "num_steps": 2e6,
            "initial_learning_steps": 30_000,
            "initial_collection_steps": 30_000,
            "collect_with_policy": False,
            "eval_interval": 25*10**3,
            "num_eval_episodes": 10,
            "grad_clip_norm": 40.0,
            "tau": 5e-3,  # Exponential averaging coefficient for target network updates, nu in the paper
            "train_steps_per_iter": 100,
            "env_steps_per_train_step": 100
        }

    return cfg