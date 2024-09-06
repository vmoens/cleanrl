# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import os
import random
import time
from collections import deque
from dataclasses import dataclass

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
from torch.distributions.categorical import Categorical, Distribution
from torch.utils.tensorboard import SummaryWriter

Distribution.set_default_validate_args(False)


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

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
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

    # env setup
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

    agent = Agent(envs, device=device)
    # Make a version of agent with detached params
    agent_inference = Agent(envs, device=device)
    tensordict.TensorDict.from_module(agent).detach().to_module(agent_inference)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)


    # ALGO Logic: Storage setup
    @tensordict.tensorclass
    class Transitions:
        obs: torch.Tensor
        actions: torch.Tensor
        logprobs: torch.Tensor
        dones: torch.Tensor
        vals: torch.Tensor
        rewards: torch.Tensor
        advantages: torch.Tensor=None
        returns: torch.Tensor=None


    container = Transitions(
        obs=torch.zeros(envs.single_observation_space.shape),
        actions=torch.zeros(envs.single_action_space.shape),
        logprobs=torch.zeros(()),
        rewards=torch.zeros(()),
        dones=torch.zeros((), dtype=torch.bool),
        vals=torch.zeros(()),
        advantages=torch.zeros(()),
        returns=torch.zeros(()),
        device=device,
    ).expand(args.num_steps, args.num_envs).clone()
    container_flat = container.view(-1)
    container_local = container_flat[:args.minibatch_size].clone()

    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs = torch.tensor(envs.reset(), device=device, dtype=torch.float)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)


    def update(container: Transitions):
        b_obs_mb_inds = container.obs
        b_actions_mb_inds = container.actions
        b_logprobs_mb_inds = container.logprobs
        b_advantages_mb_inds = container.advantages
        b_returns_mb_inds = container.returns
        b_values_mb_inds = container.vals

        optimizer.zero_grad()
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs_mb_inds, b_actions_mb_inds)
        logratio = newlogprob - b_logprobs_mb_inds
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

        mb_advantages = b_advantages_mb_inds
        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if args.clip_vloss:
            v_loss_unclipped = (newvalue - b_returns_mb_inds) ** 2
            v_clipped = b_values_mb_inds + torch.clamp(
                newvalue - b_values_mb_inds,
                -args.clip_coef,
                args.clip_coef,
            )
            v_loss_clipped = (v_clipped - b_returns_mb_inds) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - b_returns_mb_inds) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()
        return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac


    def gae(next_obs, next_done, container):
        # bootstrap value if not done
        next_value = get_value(next_obs).reshape(-1)
        lastgaelam = 0
        advantages = []
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = (~next_done).float()
                nextvalues = next_value
            else:
                nextnonterminal = (~container.dones[t + 1]).float()
                nextvalues = container.vals[t + 1]
            delta = container.rewards[t] + args.gamma * nextvalues * nextnonterminal - container.vals[t]
            advantages.append(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
            lastgaelam = advantages[-1]
        container.advantages = torch.stack(list(reversed(advantages)))
        container.returns = container.advantages + container.vals


    def rollout(global_step, obs, done):
        ts = []
        for step in range(0, args.num_steps):
            global_step += args.num_envs

            # ALGO LOGIC: action logic
            action, logprob, _, value = policy(obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())

            ts.append(Transitions(
                obs=obs,
                dones=done,
                vals=value.flatten(),
                actions=action,
                logprobs=logprob,
                rewards=reward.reshape(-1),
                device=device,
                batch_size=(args.num_envs,)
            ))

            next_obs, next_done = torch.tensor(next_obs, dtype=torch.float).to(device=device, non_blocking=True), torch.tensor(next_done, dtype=torch.bool).to(device=device, non_blocking=True)
            obs, done = next_obs, next_done

            for idx, d in enumerate(next_done):
                if d and info["lives"][idx] == 0:
                    avg_returns.append(info["r"][idx])
                    # TODO
                    # print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    # writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    # writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    # writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # torch.stack(ts, 0, out=container)
        new_container = torch.stack(ts, 0)
        gae(next_obs, next_done, new_container)
        return global_step, next_obs, next_done, new_container


    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value
    if args.compile or args.cudagraphs:
        args.compile = True
        update = torch.compile(update)
        # policy = torch.compile(policy, mode="reduce-overhead")
        # get_value = torch.compile(get_value, mode="reduce-overhead")
        rollout = torch.compile(rollout, mode="reduce-overhead")
        if args.cudagraphs:
            # graph_policy = torch.cuda.CUDAGraph()
            graph_update = torch.cuda.CUDAGraph()

    b_obs_mb_inds = None
    b_logprobs_mb_inds = None
    b_actions_mb_inds = None
    b_advantages_mb_inds = None
    b_returns_mb_inds = None
    b_values_mb_inds = None

    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    global_step_burnin = None
    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()
        # TODO
        # Annealing the rate if instructed to do so.
        # if args.anneal_lr:
        #     frac = 1.0 - (iteration - 1.0) / args.num_iterations
        #     lrnow = frac * args.learning_rate
        #     optimizer.param_groups[0]["lr"] = lrnow

        global_step, next_obs, next_done, container = rollout(global_step, next_obs, next_done)
        container_flat = container.view(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                container_local.update_(container_flat[mb_inds])

                if not args.cudagraphs or (iteration == 1 and epoch == 0 and start == 0):
                    # Run a first time without capture
                    approx_kl, v_loss, pg_loss, entropy_loss, old_approx_kl, clipfrac = update(container_local)
                elif iteration == 1 and epoch == 0 and start == args.minibatch_size and args.cudagraphs:
                    # Run a second time with capture
                    with torch.cuda.graph(graph_update):
                        approx_kl, v_loss, pg_loss, entropy_loss, old_approx_kl, clipfrac = update(container_local)
                else:
                    # Run captured graph
                    graph_update.replay()

                # TODO
                # clipfracs += [clipfrac.clone()]

            # TODO
            # if args.target_kl is not None and approx_kl > args.target_kl:
            #     break

        # TODO
        # y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        # var_y = np.var(y_true)
        # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TODO
        # # TRY NOT TO MODIFY: record rewards for plotting purposes
        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", torch.stack(clipfracs).mean(), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if global_step_burnin is not None:
            pbar.set_description(f"speed: {(global_step - global_step_burnin) / (time.time() - start_time): 4.4f} sps")
        # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
