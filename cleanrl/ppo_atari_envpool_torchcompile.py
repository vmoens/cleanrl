# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import envpool
# import gymnasium as gym
import gym
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
from tensordict import from_module
from tensordict.nn import TensorDictModule
from torch.distributions.categorical import Categorical, Distribution
from torch.utils.tensorboard import SummaryWriter
from tensordict.utils import timeit

Distribution.set_default_validate_args(False)
torch.set_float32_matmul_precision('high')

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    compile: bool = False
    """whether to use compile"""
    cudagraphs: bool = False
    """whether to use cudagraphs"""
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
    # env_id: str = "Breakout-v5"
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""


class CudaGraphCompiledModule:
    def __init__(self, module, warmup=2):
        self.module = module
        self.counter = 0
        self.warmup = warmup
        if hasattr(module, "in_keys"):
            self.in_keys = module.in_keys
        if hasattr(module, "out_keys"):
            self.out_keys = module.out_keys

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

class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, device=None):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4, device=device)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2, device=device)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1, device=device)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512, device=device)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n, device=device), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1, device=device), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, obs, action=None):
        hidden = self.network(obs / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = tyro.cli(Args)
    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
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

    ####### Environment setup #######
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Register step as a special op not to graph break
    # @torch.library.custom_op("mylib::step", mutates_args=())
    def step_func(action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with timeit("rollout - step - env"):
            next_obs_np, reward, next_done, info = envs.step(action.cpu().numpy())
        return torch.as_tensor(next_obs_np).to(device, non_blocking=True), torch.as_tensor(reward).to(device, non_blocking=True), torch.as_tensor(next_done).to(device, non_blocking=True), info

    # @step_func.register_fake
    # def _(action):
    #     return (torch.empty((args.num_envs, 4, 84, 84), dtype=torch.uint8),
    #             torch.empty((args.num_envs,), dtype=torch.float),
    #             torch.empty((args.num_envs,), dtype=torch.bool))

    ####### Agent #######
    agent = Agent(envs, device=device)
    # Make a version of agent with detached params
    agent_inference = Agent(envs, device=device)
    from_module(agent).detach().to_module(agent_inference)

    ####### Optimizer #######
    optimizer = optim.Adam(agent.parameters(), lr=torch.tensor(args.learning_rate, device=device), eps=1e-5)

    ####### Executables #######
    # Define networks: wrapping the policy in a TensorDictModule allows us to use CudaGraphCompiledModule
    policy = TensorDictModule(agent_inference.get_action_and_value, in_keys=["obs"], out_keys=["action", "log_prob", "entropy", "value"])
    get_value = agent_inference.get_value

    # Compile policy
    if args.compile or args.cudagraphs:
        args.compile = True
        policy = torch.compile(policy)
        if args.cudagraphs:
            policy = CudaGraphCompiledModule(policy)

    def gae(next_obs, next_done, container):
        # bootstrap value if not done
        next_value = get_value(next_obs).reshape(-1)
        lastgaelam = 0
        dones = container["dones"].unbind(0)
        vals = container["vals"].unbind(0)
        rewards = container["rewards"].unbind(0)

        advantages = []
        for t in range(args.num_steps - 1, -1, -1):
            if t == args.num_steps - 1:
                nextnonterminal = (~next_done).float()
                nextvalues = next_value
            else:
                nextnonterminal = (~dones[t + 1]).float()
                nextvalues = vals[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - vals[t]
            advantages.append(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
            lastgaelam = advantages[-1]
        container["advantages"] = torch.stack(list(reversed(advantages)))
        container["returns"] = container["advantages"] + container["vals"]
        return container

    if args.compile:
        gae = torch.compile(gae, fullgraph=True)
        if args.cudagraphs:
            gae = CudaGraphCompiledModule(TensorDictModule(gae, in_keys=["next_obs", "next_done", "container"], out_keys=["container"]))

    def rollout(obs, done, get_returns=False, avg_returns=[]):
        ts = []
        for step in range(args.num_steps):
            # ALGO LOGIC: action logic
            with timeit("rollout - policy"):
                action, logprob, _, value = policy(obs=obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            with timeit("rollout - step"):
                next_obs, reward, next_done, info = step_func(action)

            if get_returns:
                idx = next_done & info["lives"] == 0
                avg_returns.append(info["r"][idx].mean())

            ts.append(
                tensordict.TensorDict._new_unsafe(
                    obs=obs,
                    dones=done,
                    vals=value.flatten(),
                    actions=action,
                    logprobs=logprob,
                    rewards=reward,
                    batch_size=(args.num_envs,)
                )
            )

            obs = next_obs

        container = torch.stack(ts, 0).to(device)
        next_done = next_done.to(device, non_blocking=True)
        return next_obs, next_done, container

    # if args.compile:
    #     rollout = torch.compile(rollout)

    def update(obs, actions, logprobs, advantages, returns, vals):
        optimizer.zero_grad()
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
        logratio = newlogprob - logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

        mb_advantages = advantages
        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if args.clip_vloss:
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = vals + torch.clamp(
                newvalue - vals,
                -args.clip_coef,
                args.clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()
        return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac

    update = tensordict.nn.TensorDictModule(
        update,
        in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
        out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac"]
    )

    if args.compile:
        update = torch.compile(update)
        if args.cudagraphs:
            update = CudaGraphCompiledModule(update)

    # Tacking variables - DO NOT CHANGE
    avg_returns = deque(maxlen=20)
    global_step = 0
    container_local = None
    next_obs = torch.tensor(envs.reset(), device=device, dtype=torch.uint8)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)


    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    global_step_burnin = None
    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            if not isinstance(optimizer.param_groups[0]["lr"], torch.Tensor):
                optimizer.param_groups[0]["lr"] = torch.tensor(optimizer.param_groups[0]["lr"], device=device)
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        torch.compiler.cudagraph_mark_step_begin()
        with timeit("rollout"):
            next_obs, next_done, container = rollout(next_obs, next_done, get_returns=iteration % 10 == 0, avg_returns=avg_returns)
        global_step += container.numel()

        with timeit("gae"):
            container = gae(next_obs, next_done, container)
        with timeit("view"):
            container_flat = container.view(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size, device=device)
            b_inds = b_inds.split(args.minibatch_size)
            for start, b in zip(range(0, args.batch_size, args.minibatch_size), b_inds):
                end = start + args.minibatch_size

                with timeit("copy_data"):
                    if container_local is None:
                        container_local = container_flat[b].clone()
                    else:
                        container_local.update_(container_flat[b])

                with timeit("update"):
                    out = update(container_local, tensordict_out=tensordict.TensorDict())

        if global_step_burnin is not None and iteration % 10 == 0:
            pbar.set_description(f"speed: {(global_step - global_step_burnin) / (time.time() - start_time): 4.1f} sps, "
                                 f"reward avg: {container['rewards'].mean():4.2f}, "
                                 f"reward max: {container['rewards'].max():4.2f}, "
                                 f"returns: {torch.tensor(avg_returns).mean(): 4.2f},"
                                 f"lr: {optimizer.param_groups[0]['lr']: 4.2f}")
            timeit.print()

    envs.close()
    writer.close()
