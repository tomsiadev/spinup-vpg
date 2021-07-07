import gym

from models import *

def test():
    env_name = 'CartPole-v1'

    actor_hidden_sizes = [64, 64]

    env = gym.make(env_name)

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    actor_net = ActorNet(obs_size, actor_hidden_sizes, act_size).float()

    actor_net.load_state_dict(torch.load("models/actor_1400"))

    obs = env.reset()
    done = False

    while not done:
        env.render()

        act = actor_net.get_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, rew, done, _ = env.step(act)

if __name__=='__main__':
    test()