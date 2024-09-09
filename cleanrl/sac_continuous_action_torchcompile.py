# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import math
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
# from stable_baselines3.common.buffers import ReplayBuffer
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import from_modules, TensorDict
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    compile: bool = False
    cudagraphs: bool = False

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, n_act, n_obs, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_act + n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc3 = nn.Linear(256, 1, device=device)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, n_obs, n_act, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc_mean = nn.Linear(256, n_act, device=device)
        self.fc_logstd = nn.Linear(256, n_act, device=device)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32, device=device)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    n_act = math.prod(envs.single_action_space.shape)
    n_obs = math.prod(envs.single_observation_space.shape)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs, device=device, n_act=n_act, n_obs=n_obs)
    def get_q_params():
        qf1 = SoftQNetwork(envs, device=device, n_act=n_act, n_obs=n_obs)
        qf2 = SoftQNetwork(envs, device=device, n_act=n_act, n_obs=n_obs)
        qf1_target = SoftQNetwork(envs, device=device, n_act=n_act, n_obs=n_obs)
        qf2_target = SoftQNetwork(envs, device=device, n_act=n_act, n_obs=n_obs)
        qnet_params = from_modules(qf1, qf2, as_module=True)
        print('qnet_params', qnet_params)
        qnet_target = from_modules(qf1_target, qf2_target).data
        qnet_target.update_(qnet_params.data)
        # discard params of net
        qnet = SoftQNetwork(envs, device="meta", n_act=n_act, n_obs=n_obs)
        qnet_params.to_module(qnet)

        return qnet_params, qnet_target, qnet

    qnet_params, qnet_target, qnet = get_q_params()
    q_optimizer = optim.Adam(qnet_params.parameters(), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = torch.as_tensor(args.alpha, device=device)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    def batched_qf(params, obs, action, next_q_value=None):
        with params.to_module(qnet):
            vals = qnet(obs, action)
            if next_q_value is not None:
                loss_val = F.mse_loss(vals, next_q_value)
                return loss_val
            return vals

    def update_main(data):
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(data["next_observations"])
            qf_next_target = torch.vmap(batched_qf, (0, None, None))(qnet_target, data["next_observations"], next_state_actions)
            min_qf_next_target = qf_next_target.min(dim=0).values - alpha * next_state_log_pi
            next_q_value = data["rewards"].flatten() + (~data["dones"].flatten()).float() * args.gamma * (min_qf_next_target).view(
                -1)

        qf_a_values = torch.vmap(batched_qf, (0, None, None, None))(qnet_params, data["observations"], data["actions"], next_q_value)
        qf_loss = qf_a_values.sum(0)

        # optimize the model
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

    def update_pol(data):
        pi, log_pi, _ = actor.get_action(data["observations"])
        qf_pi = torch.vmap(batched_qf, (0, None, None))(qnet_params.data, data["observations"], pi)
        min_qf_pi = qf_pi.min(0).values
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        if args.autotune:
            with torch.no_grad():
                _, log_pi, _ = actor.get_action(data["observations"])
            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

            a_optimizer.zero_grad()
            alpha_loss.backward()
            a_optimizer.step()
        return alpha

    if args.compile:
        update_main = torch.compile(update_main)
        update_pol = torch.compile(update_pol)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    for global_step in pbar:
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                desc  = f"global_step={global_step}, episodic_return={info['episode']['r']}"
                # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        transition = TensorDict(
            observations=torch.as_tensor(obs, device=device, dtype=torch.float),
            next_observations=torch.as_tensor(real_next_obs, device=device, dtype=torch.float),
            actions=torch.as_tensor(actions, device=device, dtype=torch.float),
            rewards=torch.as_tensor(rewards, device=device, dtype=torch.float),
            terminations=terminations,
            dones=terminations,
            batch_size=obs.shape[0],
            device=device
        )
        rb.extend(transition)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            update_main(data)
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    update_pol(data)
                    alpha.copy_(log_alpha.detach().exp())

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                qnet_target.lerp_(qnet_params.data, args.tau)

            if global_step % 100 == 0:
                # writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                # writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                # writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                # writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                # writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                # writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                # writer.add_scalar("losses/alpha", alpha, global_step)
                if start_time is not None:
                    pbar.set_description(f"{(global_step - measure_burnin) / (time.time() - start_time): 4.4f} sps, "+desc)
                # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                # if args.autotune:
                #     writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()
