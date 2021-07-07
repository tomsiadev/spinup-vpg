import torch
import numpy as np

def compute_policy_loss(actor_net, obs, act, adv):
    log_prob = actor_net.get_policy(obs, True).log_prob(act)
    return -(log_prob * adv).mean()


def compute_value_loss(value_net, obs, rtg):
    val = value_net.get_value(obs, True)
    return ((val - rtg) ** 2).mean()


def reward_to_go(reward_list):
    reward_len = len(reward_list)
    rtg = [0] * reward_len
    accum = 0

    for i in reversed(range(reward_len)):
        accum += reward_list[i]
        rtg[i] = accum

    return rtg


def compute_TD(value_net, obs, next_obs, reward, done, gamma):
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32)

    next_val = gamma * value_net.get_value(next_obs_tensor) if not done else 0
    return (reward + next_val - value_net.get_value(obs_tensor))#.detach().numpy()


def compute_GAE(TD_list, gamma, lambd):

    GAE = []

    for i in range(len(TD_list)):
        accum = 0
        factor = 1
        for _ in range(i, len(TD_list)):
            accum += factor * TD_list[i]
            factor *= gamma * lambd
        GAE.append(accum)

    # GAE = np.array(GAE)
    #
    # if len(GAE) > 1:
    #     gae_mean = np.mean(GAE)
    #     gae_std = np.std(GAE)
    #     GAE = (GAE - gae_mean) / gae_std
    # else:
    #     return np.zeros(GAE.shape)

    return GAE


class Trajectory:
    def __init__(self, gamma, lambd):
        self.obs = []
        self.acts = []
        self.rewards = []
        self.TDs = []

        self.gamma = gamma
        self.lambd = lambd

    def compute_adv_rtg(self):
        adv = compute_GAE(self.TDs, self.gamma, self.lambd)
        rtg = reward_to_go(self.rewards)

        return adv, rtg


class EpochBuffer:
    def __init__(self):
        self.obs = []
        self.acts = []
        self.adv = []
        self.rtg = []

        self.episodes = 0
        self.total_episodes_length = 0

    def append_trajectory(self, trajectory):
        self.obs.extend(trajectory.obs)
        self.acts.extend(trajectory.acts)

        adv, rtg = trajectory.compute_adv_rtg()

        self.adv.extend(adv)
        self.rtg.extend(rtg)

        self.episodes += 1
        self.total_episodes_length += len(trajectory.obs)


def train_one_epoch(env, actor_net, value_net, actor_optim, value_optim, gamma, lambd,
                    value_updates_per_trajectory, max_epoch_length=5000, render=True):

    obs = env.reset()

    epoch_buffer = EpochBuffer()

    curr_trajectory = Trajectory(gamma, lambd)

    for i in range(max_epoch_length):

        if render:
            env.render()

        curr_trajectory.obs.append(obs)

        act = actor_net.get_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, rew, done, _ = env.step(act)

        curr_trajectory.acts.append(act)
        curr_trajectory.rewards.append(rew)

        TD = compute_TD(value_net, curr_trajectory.obs[-1], obs, rew, done, gamma)
        curr_trajectory.TDs.append(TD)

        if done:

            epoch_buffer.append_trajectory(curr_trajectory)

            curr_trajectory = Trajectory(gamma, lambd)
            obs = env.reset()

    if len(curr_trajectory.obs) > 0:
        epoch_buffer.append_trajectory(curr_trajectory)

    update(epoch_buffer, actor_net, value_net, actor_optim, value_optim, value_updates_per_trajectory)


def update(epoch_buffer, actor_net, value_net, actor_optim, value_optim, value_updates_per_trajectory):

    obs_tensor = torch.as_tensor(epoch_buffer.obs, dtype=torch.float32)
    acts_tensor = torch.as_tensor(epoch_buffer.acts, dtype=torch.int32)
    adv_tensor = torch.as_tensor(epoch_buffer.adv, dtype=torch.float32)
    rtg_tensor = torch.as_tensor(epoch_buffer.rtg, dtype=torch.float32)

    actor_optim.zero_grad()
    policy_loss = compute_policy_loss(actor_net, obs_tensor, acts_tensor, adv_tensor)
    policy_loss.backward()
    actor_optim.step()

    for i in range(value_updates_per_trajectory):
        value_optim.zero_grad()
        value_loss = compute_value_loss(value_net, obs_tensor, rtg_tensor)
        value_loss.backward()
        value_optim.step()

    print("policy_loss: ", policy_loss, "\tvalue loss: ", value_loss,
          "\taverage episode length: ", epoch_buffer.total_episodes_length / epoch_buffer.episodes)
