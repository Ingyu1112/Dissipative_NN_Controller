import os
import ptan
import time 
import gymnasium as gym
import argparse
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np

from lib import model, common, custom_model

import torch
import torch.optim as optim
import torch.nn.functional as F

GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
BIAS = False

TEST_ITERS = 1000

# perform periodical tests of our model on the separate testing environment
# we don't need to do any exploration
# we will just use the mean value returned by the model directly, without any random sampling
def test_net(net: model.DDPGActor, env: gym.Env, count: int = 10,
             device: torch.device = torch.device("cpu")):
    rewards = 0.0
    steps = 0
    lqr_penalty = 0.0
    cost_q = 0.0
    cost_v = 0.0
    cost_u = 0.0
    for _ in range(count):
        obs, _ = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, is_tr, info = env.step(action)
            rewards += reward
            steps += 1

            lqr_penalty += info.get("lqr_penalty", 0.0)
            cost_q += info.get("cost_q", 0.0)
            cost_v += info.get("cost_v", 0.0)
            cost_u += info.get("cost_u", 0.0)

            if done or is_tr:
                break
    return (rewards / count, steps / count, 
            lqr_penalty / count, cost_q / count, 
            cost_v / count, cost_u / count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu",
                        help="Device to use, default=cpu")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device(args.dev)

    save_path = os.path.join("saves", "ddpg-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    custom_model.register_env()
    env = gym.make(custom_model.ENV_ID)
    test_env = gym.make(custom_model.ENV_ID)

    act_net = model.DDPGActor(env.observation_space.shape[0],
                              env.action_space.shape[0], bias).to(device)
    crt_net = model.DDPGCritic(env.observation_space.shape[0],
                               env.action_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="-ddpg_" + args.name)
    agent = model.AgentDDPG(act_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)    # use two different optimizoers to simplify
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)    # the way we handle gradients for the actor and acritic training steps

    frame_idx = 0
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(
                writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v = \
                    common.unpack_batch_ddpg(batch, device)

                # train critic
                crt_opt.zero_grad()
                q_v = crt_net(states_v, actions_v)  # Q value estimation
                last_act_v = tgt_act_net.target_model(last_states_v)            # off policy method
                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)  # off policy method
                q_last_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA    # real Q value from off policy
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())   # critic loss
                critic_loss_v.backward()
                crt_opt.step()  # SGD
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                actor_loss_v = -crt_net(states_v, cur_actions_v)    # actor loss
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()  #SGD, only touch the actor's optimizer to do the optimization step
                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                # update target network
                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps, lqr, q, v, u = test_net(act_net, test_env, device=device)
                    print("Test done in %.2f sev, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)
                    writer.add_scalar("lqr/lqr_penalty", lqr, frame_idx)
                    writer.add_scalar("lqr/cost_q", q, frame_idx)
                    writer.add_scalar("lqr/cost_v", v, frame_idx)
                    writer.add_scalar("lqr/cost_u", u, frame_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)

                            # save video
                            video_env = gym.make(custom_model.ENV_ID, render_mode="rgb_array")
                            video_env = gym.wrappers.RecordVideo(
                                video_env, 
                                video_folder=os.path.join(save_path, "videos"), 
                                name_prefix=f"best_step_{frame_idx}_reward_{rewards:.1f}"
                            )
                            test_net(act_net, video_env, count=1, device=device)
                            video_env.close()
                        best_reward = rewards

                if best_reward is not None and best_reward > 500:
                    print("Successfully cleared! Reward: %.2f" % best_reward)
                    break
                     







