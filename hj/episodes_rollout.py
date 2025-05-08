import gymnasium as gym
import torch, numpy as np, torch.nn as nn
import tianshou as ts
from formularone import SafeGymnasium
import os
import sys
import ruamel.yaml as yaml
sys.path.append("D:/Code/WashU/vision_project/dreamerv3-torch/")
from dreamer_nom import Dreamer
import pathlib
import argparse
import tools

parser = argparse.ArgumentParser()
parser.add_argument("--configs", nargs="+")
configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

name_list = ["defaults", "safe_gymnasium"]
defaults = {}
for name in name_list:
    recursive_update(defaults, configs[name])
parser = argparse.ArgumentParser()
for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = tools.args_type(value)
    parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
config = parser.parse_args()
# train_envs = SafeGymnasium(name="SafetyCarFormulaOne1-v0")
# train_envs = SafeGymnasium(name="SafetyCarFormulaOne1Vision-v0")
train_envs = SafeGymnasium(name="SafetyCarGoal2Vision-v0")
acts = train_envs.action_space
config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
agent = Dreamer(
        train_envs.observation_space,
        train_envs.action_space,
        config,
        None,
        None,
    ).to(config.device)
state_dict = torch.load("cost.pt")
new_state_dict = {}
for key in list(state_dict["agent_state_dict"].keys()):
    if 'orig_mod.' in key:
        deal_key = key.replace('_orig_mod.', '')
        new_state_dict[deal_key] = state_dict["agent_state_dict"][key]
agent.load_state_dict(new_state_dict)

step, episode = 0, 0
done = np.ones(1, bool)
length = np.zeros(1, np.int32)
obs = [None] * 1
agent_state = None
reward = [0] * 1

imgs = []
vectors = []
features = []
agent_latent = []
embs = []
values = []
actions = []
costs = []
from tqdm import trange
for _ in trange(100):
    cost = 0
    obs = train_envs.reset()
    obs = {k: np.expand_dims(np.array(obs[k]),axis=0) for k in obs}
    action, agent_state, embed = agent(obs, agent_state)
    (latent, _) = agent_state
    feat = agent._wm.dynamics.get_feat(latent)
    value = agent._task_behavior.value(feat).mode()
    action = action["action"].cpu().numpy()
    episode_imgs = []
    episode_vectors = []
    episode_values = []
    episode_actions = []
    episode_costs = []
    episode_features = []
    episode_latent = []
    episode_embs = []
    for i in range(1000):
        obs, reward, done, info = train_envs.step(action[0])
        obs = {k: np.expand_dims(np.array(obs[k]),axis=0) for k in obs}
        action, agent_state, embed = agent(obs, agent_state) #agent_state = ( latent , prev_action )
        (latent, _) = agent_state ## latent contains h and z

## 1) below is executed when we call agent(obs, agent state)
#     def _policy(self, obs, state, training):
#         if state is None:
#             latent = action = None
#         else:
#             latent, action = state
#         obs = self._wm.preprocess(obs)
# #This is Zt encoded, pure perceptual representation of the raw image and state vector
#         embed = self._wm.encoder(obs) 
# #It uses the previous latent and previous action to generate a prior p(zt)
# #Then, it corrects this with the current embed (the encoded x(t) which is the embed) to sample from a posterior q(zt/ht prior, embed)->zt posterior
#         latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
#         if self._config.eval_state_mean:
#             latent["stoch"] = latent["mean"]
#         feat = self._wm.dynamics.get_feat(latent)
#         if not training:
#             actor = self._task_behavior.actor(feat)
#             action = actor.mode()
#         elif self._should_expl(self._step):
#             actor = self._expl_behavior.actor(feat)
#             action = actor.sample()
#         else:
#             actor = self._task_behavior.actor(feat)
# ##imp note the actor takes feat as input meaning concat of z and h
#             action = actor.sample()
#         logprob = actor.log_prob(action)
#         latent = {k: v.detach() for k, v in latent.items()}
#         action = action.detach()
#         if self._config.actor["dist"] == "onehot_gumble":
#             action = torch.one_hot(
#                 torch.argmax(action, dim=-1), self._config.num_actions
#             )
#         policy_output = {"action": action, "logprob": logprob}
#         state = (latent, action)
#         return policy_output, state, embed


​

# 2)
    # def get_feat(self, state):
        # stoch = state["stoch"]
        # if self._discrete:
        #     shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
        #     stoch = stoch.reshape(shape)
        # return torch.cat([stoch, state["deter"]], -1)
        
# where 
#     "deter" → deterministic RNN state h(t)
#     "stoch" → stochastic state sample z(t)

        feat = agent._wm.dynamics.get_feat(latent)
        value = agent._task_behavior.value(feat).mode()
        feat = feat.detach().cpu().numpy()
        action = action["action"].cpu().numpy()
        episode_actions.append(action[0])
        episode_imgs.append(obs["image"][0])
        episode_vectors.append(obs["vector"][0])
        episode_features.append(feat[0])
        episode_values.append(value.detach().cpu().numpy()[0])
        episode_costs.append(reward)
        episode_latent.append(latent)
        episode_embs.append(embed.detach().cpu().numpy()[0])
    imgs = imgs + episode_imgs[:-1]
    vectors = vectors + episode_vectors[:-1]
    values = values + episode_values[:-1]
    actions = actions + episode_actions[:-1]
    features = features + episode_features[:-1]
    costs = costs + episode_costs[1:]
    agent_latent = agent_latent + episode_latent[:-1]
    embs = embs + episode_embs[:-1]

imgs = np.array(imgs)
vectors = np.array(vectors)
values = np.array(values)
actions = np.array(actions)
features = np.array(features)
embs = np.array(embs)
agent_latent = {k: np.array([agent_latent[i][k][0].detach().cpu().numpy() for i in range(len(agent_latent))]) for k in agent_latent[0].keys()}
costs = np.array(costs).reshape(-1, 1)
s = {"img":imgs, "vector": vectors, "value": values, "latent": agent_latent, "emb": embs}
print(imgs.shape, vectors.shape, values.shape, features.shape, embs.shape)
for k in agent_latent.keys():
    print(k, agent_latent[k].shape)
print(actions.shape, costs.shape)
import pickle
with open("episodes100.pkl", "wb") as f:
    pickle.dump((s, actions, costs), f)