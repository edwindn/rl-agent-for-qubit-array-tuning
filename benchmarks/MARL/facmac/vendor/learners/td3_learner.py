"""
TD3-style learner for the baseline tuning (M1).

Diff vs MADDPGLearner:
  1. Twin critics Q1 / Q2 — target uses min(Q1', Q2') (clipped double-Q).
  2. Target policy smoothing — target actions get clipped Gaussian noise before
     evaluating target critics (reduces target overestimation, smooths Q).
  3. Delayed actor updates — actor + target nets update every `policy_delay`
     critic updates (default 2).

Hyperparameters (read from args, fall back to standard TD3 defaults):
  td3_target_policy_noise: 0.2     (sigma of target action smoothing)
  td3_target_policy_clip:  0.5     (clip range for target noise)
  td3_policy_delay:        2       (actor update frequency)

Drop-in replacement: register as 'td3_learner' and use in yaml.
"""

import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.maddpg import MADDPGCritic
import torch as th
from torch.optim import RMSprop, Adam


class TD3Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        # Twin critics
        self.critic1 = MADDPGCritic(scheme, args)
        self.critic2 = MADDPGCritic(scheme, args)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())

        opt_name = getattr(self.args, "optimizer", "rmsprop")
        if opt_name == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr,
                                           alpha=args.optim_alpha, eps=args.optim_eps)
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr,
                                            alpha=args.optim_alpha, eps=args.optim_eps)
        elif opt_name == "adam":
            eps = getattr(args, "optimizer_epsilon", 10e-8)
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr, eps=eps)
            self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr, eps=eps)
        else:
            raise Exception(f"unknown optimizer {opt_name}")

        # TD3-specific
        self.target_policy_noise = float(getattr(args, "td3_target_policy_noise", 0.2))
        self.target_policy_clip = float(getattr(args, "td3_target_policy_clip", 0.5))
        self.policy_delay = int(getattr(args, "td3_policy_delay", 2))
        self._update_count = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Build target actions WITH policy smoothing
        target_actions = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_target_outs = self.target_mac.select_actions(
                batch, t_ep=t, t_env=None, test_mode=True,
                critic=self.target_critic1, target_mac=True,
            )
            # Smooth: add clipped gaussian noise to target action.
            noise = (th.randn_like(agent_target_outs) * self.target_policy_noise
                     ).clamp(-self.target_policy_clip, self.target_policy_clip)
            agent_target_outs = (agent_target_outs + noise).clamp(-1.0, 1.0)
            target_actions.append(agent_target_outs)
        target_actions = th.stack(target_actions, dim=1)

        # Q-taken (both critics)
        q1_taken, q2_taken = [], []
        for t in range(batch.max_seq_length - 1):
            inputs = self._build_inputs(batch, t=t)
            q1, _ = self.critic1(inputs, actions[:, t:t+1].detach())
            q2, _ = self.critic2(inputs, actions[:, t:t+1].detach())
            q1_taken.append(q1.view(batch.batch_size, -1, 1))
            q2_taken.append(q2.view(batch.batch_size, -1, 1))
        q1_taken = th.stack(q1_taken, dim=1)
        q2_taken = th.stack(q2_taken, dim=1)

        # Target = min(Q1', Q2') (clipped double-Q).
        target_vals = []
        for t in range(1, batch.max_seq_length):
            target_inputs = self._build_inputs(batch, t=t)
            tq1, _ = self.target_critic1(target_inputs, target_actions[:, t:t+1].detach())
            tq2, _ = self.target_critic2(target_inputs, target_actions[:, t:t+1].detach())
            tq = th.min(tq1, tq2).view(batch.batch_size, -1, 1)
            target_vals.append(tq)
        target_vals = th.stack(target_vals, dim=1)

        q1_taken = q1_taken.view(batch.batch_size, -1, 1)
        q2_taken = q2_taken.view(batch.batch_size, -1, 1)
        target_vals = target_vals.view(batch.batch_size, -1, 1)
        targets = (rewards.expand_as(target_vals)
                   + self.args.gamma * (1 - terminated.expand_as(target_vals)) * target_vals)

        td_error1 = q1_taken - targets.detach()
        td_error2 = q2_taken - targets.detach()
        mask_e = mask.expand_as(td_error1)
        loss1 = ((td_error1 * mask_e) ** 2).sum() / mask_e.sum()
        loss2 = ((td_error2 * mask_e) ** 2).sum() / mask_e.sum()
        critic_loss = loss1 + loss2

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        self._update_count += 1

        # Delayed actor + target update
        do_actor_update = (self._update_count % self.policy_delay == 0)

        if do_actor_update:
            mac_out = []
            chosen_action_qvals = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t=t, select_actions=True)["actions"].view(
                    batch.batch_size, self.n_agents, self.n_actions)
                for idx in range(self.n_agents):
                    tem_joint_act = actions[:, t:t+1].detach().clone().view(
                        batch.batch_size, -1, self.n_actions)
                    tem_joint_act[:, idx] = agent_outs[:, idx]
                    # Use Q1 only for actor update (standard TD3).
                    q, _ = self.critic1(self._build_inputs(batch, t=t), tem_joint_act)
                    chosen_action_qvals.append(q.view(batch.batch_size, -1, 1))
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)
            chosen_action_qvals = th.stack(chosen_action_qvals, dim=1)
            pi = mac_out

            pg_loss = -chosen_action_qvals.mean() + (pi**2).mean() * 1e-3

            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

            update_mode = getattr(self.args, "target_update_mode", "hard")
            if update_mode == "hard":
                self._update_targets()
            elif update_mode in ("soft", "exponential_moving_average"):
                self._update_targets_soft(tau=getattr(self.args, "target_update_tau", 0.001))
            else:
                raise Exception(f"unknown target update mode: {update_mode}")
        else:
            agent_grad_norm = th.tensor(0.0)
            pg_loss = th.tensor(0.0)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm",
                                 critic_grad_norm.item() if hasattr(critic_grad_norm, "item")
                                 else float(critic_grad_norm), t_env)
            mask_elems = mask_e.sum().item()
            self.logger.log_stat("td_error_abs",
                                 (td_error1 * mask_e).abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("q_taken_mean", (q1_taken * mask_e).sum().item() / mask_elems, t_env)
            self.logger.log_stat("target_mean", targets.sum().item() / mask_elems, t_env)
            self.logger.log_stat("pg_loss", pg_loss.item() if hasattr(pg_loss, "item")
                                 else float(pg_loss), t_env)
            self.logger.log_stat("agent_grad_norm",
                                 agent_grad_norm.item() if hasattr(agent_grad_norm, "item")
                                 else float(agent_grad_norm), t_env)
            self.log_stats_t = t_env

    def _update_targets_soft(self, tau):
        for tp, p in zip(self.target_mac.parameters(), self.mac.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)
        for tp, p in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)
        for tp, p in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)
        if self.args.verbose:
            self.logger.console_logger.info(f"Updated all target networks (soft tau={tau})")

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = [batch["state"][:, t]]
        if self.args.recurrent_critic:
            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions"][:, t]))
                else:
                    inputs.append(batch["actions"][:, t - 1])
        return th.cat([x.reshape(bs, -1) for x in inputs], dim=1)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.logger.console_logger.info("Updated all target networks")

    def cuda(self, device="cuda:0"):
        self.mac.cuda(device=device)
        self.target_mac.cuda(device=device)
        self.critic1.cuda(device=device)
        self.critic2.cuda(device=device)
        self.target_critic1.cuda(device=device)
        self.target_critic2.cuda(device=device)

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.agent_optimiser.state_dict(), f"{path}/opt.th")

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.agent_optimiser.load_state_dict(
            th.load(f"{path}/opt.th", map_location=lambda storage, loc: storage))
