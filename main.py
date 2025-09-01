from simpleRL import run_to_completion, make_atari_env


import torch
import torch.nn as nn
from dataset import ZippedDataset, ReachabilityDataset, RewardDataset
from agent import Embedder, Evaluator, TransitionModel, SimpleTransitionModel, RewardPredictor
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn.functional as F
env_id = "ALE/Pong-v5"
env = make_atari_env(env_id, seed=42)()  # Create an environment instance
embedding_dim = 128

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

def random_policy(obs_batch):
    B = obs_batch.shape[0]
    return torch.randint(0, env.action_space.n, (B,), device=obs_batch.device)

S, A, R, TR = run_to_completion(env_fn=lambda: env, policy=random_policy, n_runs=20, device="cpu")

def sum_rewards(rewards):
    return sum(sum(ep_rewards) for ep_rewards in rewards)

print("Total rewards over random policy:", sum_rewards(R))

reward_datamodule = RewardDataset(S, R, device="cuda")
reachability_datamodule = ReachabilityDataset(S, A, device="cuda")
datamodule = ZippedDataset(reward_datamodule, reachability_datamodule)

print("Dataset size (number of (state, action) pairs):", len(datamodule))

emb = Embedder(env.observation_space.shape,embedding_dim).to("cuda")
eval = Evaluator(embedding_dim).to("cuda")
transition_model = TransitionModel(env.action_space.n, embedding_dim).to("cuda")
simple_tm = SimpleTransitionModel(embedding_dim).to("cuda")
reward_predictor = RewardPredictor(embedding_dim).to("cuda")

all_params = list(emb.parameters()) + list(eval.parameters()) + list(transition_model.parameters()) + list(simple_tm.parameters()) + list(reward_predictor.parameters())
opt = torch.optim.Adam(all_params, lr=1e-4)

ma_emb_loss = 0.0
ma_trans_loss = 0.0
ma_trans_simple_loss = 0.0
ma_emb_mean = 0.0
ma_emb_var = 0.0
ma_reward_loss = 0.0

mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()

for i in range(100):
    dataloader = torch.utils.data.DataLoader(datamodule, batch_size=512, shuffle=True)

    all_labels = []
    all_probs = []
    for reward_batch,reachability_batch in dataloader:
        opt.zero_grad()

        reward_state = reward_batch["state"].to("cuda")
        rewards = reward_batch["reward"].to("cuda")
        reward_emb = emb(reward_state)
        reward_pred = reward_predictor(reward_emb).squeeze(-1)
        reward_loss = mse_loss(reward_pred, rewards)

        reach_states = reachability_batch["state"].to("cuda")
        next_states = reachability_batch["next_state"].to("cuda")
        actions = reachability_batch["action"].to("cuda")

        comp_state = reachability_batch["compare_state"].to("cuda")
        labels = reachability_batch["label"].to("cuda")

        emb_out = emb(reach_states)

        emb_next = emb(next_states)
        emb_comp = emb(comp_state)

        eval_out = eval(emb_comp, emb_out)
        emb_loss = bce_loss(eval_out, labels)

        emb_trans = transition_model(emb_out, actions)

        trans_loss = mse_loss(emb_trans, emb_next)
        trans_simple_loss = mse_loss(simple_tm(emb_out.detach()), emb_next.detach())

        loss = emb_loss + trans_loss + trans_simple_loss + reward_loss
        loss.backward()
        opt.step()

        # moving averages
        ma_emb_loss = 0.99 * ma_emb_loss + 0.01 * emb_loss.item() if ma_emb_loss > 0 else emb_loss.item()
        ma_trans_loss = 0.99 * ma_trans_loss + 0.01 * trans_loss.item() if ma_trans_loss > 0 else trans_loss.item()
        ma_trans_simple_loss = 0.99 * ma_trans_simple_loss + 0.01 * trans_simple_loss.item() if ma_trans_simple_loss > 0 else trans_simple_loss.item()
        ma_emb_mean = 0.99 * ma_emb_mean + 0.01 * emb_out.mean().item() if ma_emb_mean > 0 else emb_out.mean().item()
        ma_emb_var = 0.99 * ma_emb_var + 0.01 * emb_out.var().item() if ma_emb_var > 0 else emb_out.var().item()
        ma_reward_loss = 0.99 * ma_reward_loss + 0.01 * reward_loss.item() if ma_reward_loss > 0 else reward_loss.item()

        # store for AUC
        probs = torch.sigmoid(eval_out).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())

    # concatenate and compute AUC after epoch
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_probs)

    print(
        f"Epoch {i:03d}: "
        f"emb_loss={ma_emb_loss:.4f}, "
        f"trans_loss={ma_trans_loss:.4f}, "
        f"trans_simple_loss={ma_trans_simple_loss:.4f}, "
        f"reward_loss={ma_reward_loss:.4f}, "
        f"emb_mean={ma_emb_mean:.4f}, "
        f"emb_var={ma_emb_var:.4f}, "
        f"AUC={auc:.4f}"
    )

#save the model
torch.save({
    'config' : {
        'env_id': env_id,
        'embedding_dim': embedding_dim,
        'n_actions': env.action_space.n,
        'obs_shape': env.observation_space.shape,
    },
    'embedder_state_dict': emb.state_dict(),
    'evaluator_state_dict': eval.state_dict(),
    'transition_model_state_dict': transition_model.state_dict(),
    'simple_tm_state_dict': simple_tm.state_dict(),
    'reward_predictor_state_dict': reward_predictor.state_dict(),
}, "model.pth")
