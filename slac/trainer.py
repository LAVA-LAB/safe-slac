import os
from collections import deque
from datetime import timedelta
from time import sleep, time
from tkinter import N

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
from tqdm import tqdm
from slac.utils import sample_reproduction
from PIL import Image
from copy import deepcopy

class SlacObservation:
    """
    Observation for SLAC.
    """

    def __init__(self, state_shape, action_shape, num_sequences):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.num_sequences = num_sequences

    def reset_episode(self, state):
        self._state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state.append(np.zeros(self.state_shape, dtype=np.uint8))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state.append(state)

    def append(self, state, action):
        self._state.append(state)
        self._action.append(action)

    @property
    def state(self):
        return np.array(self._state)[None, ...]
    
    @property
    def last_state(self):
        return np.array(self._state[-1])[None, ...]

    @property
    def action(self):
        return np.array(self._action).reshape(1, -1)

    @property
    def last_action(self):
        return np.array(self._action[-1])


class Trainer:
    """
    Trainer for SLAC.
    """

    def __init__(
        self,
        env,
        env_test,
        algo,
        log_dir,
        seed=0,
        num_steps=3 * 10 ** 6,
        initial_collection_steps=2* 10 ** 4,
        initial_learning_steps=10 ** 4,
        collect_with_policy=False,
        num_sequences=8,
        eval_interval=1*10 ** 3,
        num_eval_episodes=3,
        env_steps_per_train_step=1,
        action_repeat=1,
        train_steps_per_iter=1
    ):
        # Env to collect samples.
        self.env = env
        self.env.seed(seed)
        self.train_steps_per_iter = train_steps_per_iter
        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2 ** 31 - seed)

        # Observations for training and evaluation.
        self.ob = SlacObservation(env.observation_space.shape, env.action_space.shape, num_sequences)
        self.ob_test = SlacObservation(env.observation_space.shape, env.action_space.shape, num_sequences)

        # Algorithm to learn.
        self.algo = algo
        # Log setting.
        self.log = {"step": [], "return": [], "cost": []}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.log_dir = log_dir
        self.summary_dir = os.path.join(log_dir, "summary")
        self.writer = SummaryWriter(log_dir=self.summary_dir, flush_secs=10)
        self.model_dir = os.path.join(log_dir, "model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.action_repeat = action_repeat
        self.num_steps = num_steps
        self.initial_collection_steps = initial_collection_steps
        self.initial_learning_steps = initial_learning_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.env_steps_per_train_step=env_steps_per_train_step
        self.collect_with_policy = collect_with_policy

    def debug_save_obs(self, state, name, step=0):
        self.writer.add_image(f"observation_{name}", state.astype(np.uint8), global_step=step)

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        self.env.unwrapped.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
        state = self.env.reset()
        self.env.unwrapped.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
        self.ob.reset_episode(state)
        self.algo.buffer.reset_episode(state)

        # Collect trajectories using random policy.
        bar = tqdm(range(1, self.initial_collection_steps + 1))
        for step in bar:
            t = self.algo.step(self.env, self.ob, t, (not self.collect_with_policy) and step <= self.initial_collection_steps, self.writer)
        # Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.
        bar = tqdm(range(self.initial_learning_steps))
        for _ in bar:
            bar.set_description("Updating latent variable model.")
            self.algo.update_latent(self.writer)

        # Iterate collection, update and evaluation.
        for step in range(self.initial_collection_steps + 1, self.num_steps // self.action_repeat + 1):
            t = self.algo.step(self.env, self.ob, t, False, self.writer)
            self.algo.update_lag(t, self.writer)
            # Update the algorithm.
            if t%self.env_steps_per_train_step==0:
                for _ in range(self.train_steps_per_iter):
                    
                    self.algo.update_latent(self.writer)
                    self.algo.update_sac(self.writer)

            # Evaluate regularly.
            step_env = step * self.action_repeat
            if step_env % self.eval_interval == 0:
                self.evaluate(step_env)
            
            if step_env%1000==0:
                for sched in self.algo.scheds:
                    sched.step()
            
            if step_env%self.algo.epoch_len == 0:
                self.writer.add_scalar("cost/train", np.mean(self.algo.epoch_costreturns), global_step=step_env)
                self.writer.add_scalar("return/train", np.mean(self.algo.epoch_rewardreturns), global_step=step_env)
                self.algo.epoch_costreturns = []
                self.algo.epoch_rewardreturns = []
            

        # Wait for logging to be finished.
        sleep(10)

    def evaluate(self, step_env):
        reward_returns = []
        cost_returns = []
        steps_until_dump_obs = 20
        def coord_to_im_(coord):
            coord = (coord+1.5)*100
            return coord.astype(int)
        
        obs_list = []
        track_list = []
        recons_list = []
        video_spf = 2//self.action_repeat
        video_fps = 25/video_spf
        render_kwargs = deepcopy(self.env_test.env._render_kwargs["pixels"])
        render_kwargs["camera_name"] = "track"
        for i in range(self.num_eval_episodes):
            self.algo.z1 = None
            self.algo.z2 = None
            self.env_test.unwrapped.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
            state = self.env_test.reset()
            self.env_test.unwrapped.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
            self.ob_test.reset_episode(state)
            self.env_test.unwrapped.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
            episode_return = 0.0
            cost_return = 0.0
            done = False
            eval_step = 0
            while not done:
            
                action = self.algo.explore(self.ob_test)
                if i == 0 and eval_step%video_spf == 0:
                    im = self.ob_test.state[0][-1].astype("uint8")
                    obs_list.append(im)
                    reconstruction = sample_reproduction(self.algo.latent, self.algo.device, self.ob_test.state, np.array([self.ob_test._action]))[0][-1]*255
                    reconstruction = reconstruction.astype("uint8")
                    recons_list.append(reconstruction)
                   
                    track = self.env_test.unwrapped.sim.render(**render_kwargs)[::-1, :, :]
                    track = np.moveaxis(track,-1,0)
                    track_list.append(track)
                if steps_until_dump_obs == 0:
                    self.debug_save_obs(self.ob_test.state[0][-1], "eval", step_env)
                    
                    reconstruction = sample_reproduction(self.algo.latent, self.algo.device, self.ob_test.state, np.array([self.ob_test._action]))[0][-1]*255
                    self.debug_save_obs(reconstruction, "eval_reconstruction", step_env)
                steps_until_dump_obs -= 1
                
                self.env_test.env.env.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
                state, reward, done, info = self.env_test.step(action)
                self.env_test.env.env.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
                cost = info["cost"]
                
                self.ob_test.append(state, action)
                episode_return += reward
                cost_return += cost

                eval_step += 1
            if i==0:
                self.writer.add_video(f"vid/eval", [np.concatenate([obs_list,recons_list,track_list], axis=3)], global_step=step_env, fps=video_fps)
            reward_returns.append(episode_return)
            cost_returns.append(cost_return)
        self.algo.z1 = None
        self.algo.z2 = None

        # Log to CSV.
        self.log["step"].append(step_env)
        mean_reward_return = np.mean(reward_returns)
        mean_cost_return = np.mean(cost_returns)
        median_reward_return = np.median(reward_returns)
        median_cost_return = np.median(cost_return)
        self.log["return"].append(mean_reward_return)
        self.log["cost"].append(mean_cost_return)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_reward_return, step_env)
        self.writer.add_scalar("return/test_median", median_reward_return, step_env)
        self.writer.add_scalar("cost/test", mean_cost_return, step_env)
        self.writer.add_scalar("cost/test_median", median_cost_return, step_env)
        self.writer.add_histogram("return/test_hist", np.array(reward_returns), step_env)
        self.writer.add_histogram("cost/test_hist", np.array(cost_returns), step_env)
        
        print(f"Steps: {step_env:<6}   " f"Return: {mean_reward_return:<5.1f} " f"CostRet: {mean_cost_return:<5.1f}   " f"Time: {self.time}")

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
