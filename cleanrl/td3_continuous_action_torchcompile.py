# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
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
from tensordict.nn import TensorDictModule
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict, from_module, from_modules
from torch.utils.tensorboard import SummaryWriter
import tensordict

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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""

    compile: bool = False
    cudagraphs: bool = False

class CudaGraphCompiledModule:
    def __init__(self, module, warmup=2, in_keys=None, out_keys=None):
        self.module = module
        self.counter = 0
        self.warmup = warmup
        if hasattr(module, "in_keys"):
            self.in_keys = module.in_keys
        else:
            self.in_keys = in_keys if in_keys is not None else []
        if hasattr(module, "out_keys"):
            self.out_keys = module.out_keys
        else:
            self.out_keys = out_keys if out_keys is not None else []


    @tensordict.nn.dispatch(auto_batch_size=False)
    def __call__(self, tensordict, *args, **kwargs):
        if self.counter < self.warmup:
            out = self.module(tensordict, *args, **kwargs)
            self.counter += 1
            return out
        elif self.counter == self.warmup:
            self.graph = torch.cuda.CUDAGraph()
            self._tensordict = tensordict
            with torch.cuda.graph(self.graph):
                out = self.module(tensordict, *args, **kwargs)
            self._out = out
            self.counter += 1
            return out
        else:
            self._tensordict.update_(tensordict)
            self.graph.replay()
            return self._out.clone() if self._out is not None else None


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
class QNetwork(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs + n_act, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc3 = nn.Linear(256, 1, device=device)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, n_obs, n_act, env, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc_mu = nn.Linear(256, n_act, device=device)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32, device=device)
        )

    def forward(self, obs):
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        obs = self.fc_mu(obs).tanh()
        return obs * self.action_scale + self.action_bias


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
    action_low, action_high = float(envs.single_action_space.low[0]), float(envs.single_action_space.high[0])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(env=envs, n_obs=n_obs, n_act=n_act, device=device)
    actor_detach = Actor(env=envs, n_obs=n_obs, n_act=n_act, device=device)
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)


    def get_params(actor):
        qf1 = QNetwork(n_obs=n_obs, n_act=n_act, device=device)
        qf2 = QNetwork(n_obs=n_obs, n_act=n_act, device=device)
        target_actor = Actor(env=envs, device="meta", n_act=n_act, n_obs=n_obs)
        actor_params = from_module(actor).data
        target_actor_params = actor_params.clone()
        target_actor_params.to_module(target_actor)

        qnet_params = from_modules(qf1, qf2, as_module=True)
        qnet_target_params = qnet_params.data.clone()

        # discard params of net
        qnet = QNetwork(n_obs=n_obs, n_act=n_act, device="meta")
        qnet_params.to_module(qnet)

        return qnet_params, qnet_target_params, qnet, actor_params, target_actor_params, target_actor

    qnet_params, qnet_target_params, qnet, actor_params, target_actor_params, target_actor = get_params(actor)

    q_optimizer = optim.Adam(qnet.parameters(), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    def batched_qf(params, obs, action, next_q_value=None):
        with params.to_module(qnet):
            vals = qnet(obs, action)
            if next_q_value is not None:
                loss_val = F.mse_loss(vals.view(-1), next_q_value)
                return loss_val
            return vals

    def update_main(data, policy_noise = args.policy_noise, noise_clip=args.noise_clip, action_scale=target_actor.action_scale):
        with torch.no_grad():
            clipped_noise = data["noise"].mul_(policy_noise).clamp_(
                -noise_clip, noise_clip
            ).mul_(action_scale)
            # clipped_noise = torch.randn_like(data["actions"], device=device).mul_(policy_noise).mul_(action_scale)
            #
            next_state_actions = (target_actor(data["next_observations"]) + clipped_noise).clamp(
                action_low, action_high
            )
            # next_state_actions = (target_actor(data["next_observations"]) + clipped_noise)
            qf_next_target = torch.vmap(batched_qf, (0, None, None))(qnet_target_params, data["next_observations"], next_state_actions)
            min_qf_next_target = qf_next_target.min(0).values
            next_q_value = data["rewards"].flatten() + (~data["dones"].flatten()).float() * args.gamma * min_qf_next_target.view(
                -1)

        qf_loss = torch.vmap(batched_qf, (0, None, None, None))(qnet_params, data["observations"], data["actions"], next_q_value)
        qf_loss = qf_loss.sum(0)

        # optimize the model
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

    def update_pol(data):
        actor_optimizer.zero_grad()
        with qnet_params.data[0].to_module(qnet):
            actor_loss = -qnet(data["observations"], actor(data["observations"])).mean()

        actor_loss.backward()
        actor_optimizer.step()


    def extend_and_sample(transition):
        rb.extend(transition)
        return rb.sample(args.batch_size)

    is_extend_compiled = False
    if args.compile or args.cudagraphs:
        args.compile = True
        update_main = torch.compile(update_main)
        update_pol = torch.compile(update_pol)
        actor_detach = torch.compile(actor_detach)
        if args.cudagraphs:
            update_main = CudaGraphCompiledModule(update_main, warmup=3)
            update_pol = CudaGraphCompiledModule(update_pol, warmup=3)
            actor_detach = CudaGraphCompiledModule(TensorDictModule(actor_detach, in_keys=["obs"], out_keys=["action"]))

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs, device=device, dtype=torch.float)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    max_ep_ret = -float("inf")

    for global_step in pbar:
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = actor_detach(obs=obs)
            actions += torch.normal(0, actor.action_scale * args.exploration_noise)
            actions = actions.clamp(action_low, action_high).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float)
        real_next_obs = next_obs.clone()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = torch.as_tensor(infos["final_observation"][idx], device=device, dtype=torch.float)
        # obs = torch.as_tensor(obs, device=device, dtype=torch.float)
        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=torch.as_tensor(actions, device=device, dtype=torch.float),
            rewards=torch.as_tensor(rewards, device=device, dtype=torch.float),
            terminations=terminations,
            dones=terminations,
            batch_size=obs.shape[0],
            device=device
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        data = extend_and_sample(transition)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                r = float(info['episode']['r'][0])
                max_ep_ret = max(max_ep_ret, r)
                desc  = f"global_step={global_step}, episodic_return={r: 4.2f} (max={max_ep_ret: 4.2f})"
                break

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # TODO: assess this
            data["noise"] = torch.randn_like(data["actions"], device=device)
            update_main(data)
            if global_step % args.policy_frequency == 0:
                update_pol(data)

                # update the target networks
                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                qnet_target_params.lerp_(qnet_params.data, args.tau)
                target_actor_params.lerp_(actor_params.data, args.tau)

            if global_step % 100 == 0:
                if start_time is not None:
                    pbar.set_description(f"{(global_step - measure_burnin) / (time.time() - start_time): 4.4f} sps, "+desc)

    envs.close()
    writer.close()
