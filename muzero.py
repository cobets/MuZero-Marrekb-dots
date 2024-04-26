from copy import deepcopy
import torch
import torch.nn.functional as F
from line_profiler_pycharm import profile

from stats import MovingAverageScore, write_to_file
from models.model import scalar_to_support
from game import Game


def cross_loss(pred, targ):
    probs = F.log_softmax(pred, dim = -1)
    return torch.sum(- targ * probs, dim = 1).mean()


def scale_gradient(x, ratio):
    return ratio * x + (1 - ratio) * x.detach()


class MuZero:
    def __init__(self, model, optimizer, buffer, mcts, value_support = 3,
                 reward_support = 1, update_target_iteration = 1000,
                 device = 'cpu',  name = 'muzero', path = 'results/',
                 update_t_iteration = 15000, min_t = 8):

        self.model = model
        self.targ_model = deepcopy(model.value_model).to(device)
        self.optimizer = optimizer
        self.buffer = buffer
        self.mcts = mcts

        self.path = path
        self.name = name

        self.device = device

        self.value_support = value_support
        self.reward_support = reward_support

        self.update_target_iteration = update_target_iteration
        self.update_t_iteration = update_t_iteration
        self.min_t = min_t

    @profile
    def train(self, env_params, env_func, count_of_actions,
              count_of_envs = 10, max_env_steps = 1e8,
              input_dimension = (4, 96, 96)):

        print('Training is starting')

        logs_score = 'iteration,steps,episode,avg_score,best_avg_score,best_score'
        logs_loss = 'iteration,steps,episode,policy,value,reward'
        score = MovingAverageScore(count = 100)
        max_iteration = int(max_env_steps / count_of_envs + 1)

        s_policy, s_value, s_reward = 0, 0, 0
        avg_score = - 1e8

        games = [Game(env_func(**env_params), i) for i in range(count_of_envs)]
        observations = [g.reset() for g in games]

        for iteration in range(max_iteration):
            observations = torch.stack(observations).to(self.device)
            observations = observations.view(-1, *input_dimension)
            
            with torch.no_grad():
                selected_actions, root_policies, _ = self.mcts.run(
                    observations, self.model, training = True,
                    value_support = self.value_support,
                    reward_support = self.reward_support
                )

            observations = []
            end_episode = False

            for env_idx in range(count_of_envs):
                game = games[env_idx]
                obs, terminal = game.step(
                    selected_actions[env_idx].item(),
                    root_policies[env_idx]
                )

                if terminal:
                    td_rewards, td_non_terminals = game.compute_td(
                        self.buffer.td_steps,
                        self.buffer.gamma
                    )

                    start_idx = self.buffer.index
                    self.buffer.store(
                        len(game.rewards),
                        torch.stack(game.states),
                        torch.tensor(game.actions).view(-1, 1),
                        torch.tensor(game.rewards).view(-1, 1),
                        torch.tensor(game.non_terminals),
                        torch.stack(game.policies),
                        td_rewards,
                        td_non_terminals
                    )

                    end_idx = self.buffer.index

                    with torch.no_grad():
                        if start_idx < end_idx:
                            self.buffer.update_targets(
                                self.targ_model,
                                start_idx,
                                end_idx,
                                self.value_support
                            )
                        else:
                            self.buffer.update_targets(
                                self.targ_model,
                                start_idx,
                                self.buffer.max_size,
                                self.value_support
                            )
                            self.buffer.update_targets(
                                self.targ_model,
                                0,
                                start_idx,
                                self.value_support
                            )

                    score.add([game.score])
                    obs = game.reset()
                    end_episode = True
                observations.append(obs)
            
            if end_episode:
                avg_score, best_score = score.mean()
                print('iteration:', iteration,
                      '\tepisode: ', score.get_count_of_episodes(),
                      '\tmoving average score: ', avg_score)
    
                if best_score:
                    print('New best avg score has been achieved', avg_score)
                    torch.save(self.model.state_dict(), self.path + self.name + '_best.pt')
                
                logs_score += '\n' + str(iteration) + ',' \
                             + str(iteration * count_of_envs) + ',' \
                             + str(score.get_count_of_episodes()) + ',' \
                             + str(avg_score) + ',' \
                             + str(score.get_best_avg_score()) + ',' \
                             + str(score.get_best_score())

            if iteration > 1000:
                self.optimizer.zero_grad()
                idx = self.buffer.get_batch()

                states = self.buffer.states[idx].to(self.device)
                hidden_states, policies, values = self.model.initial_inference(states)

                policy_loss = cross_loss(
                    policies, self.buffer.policies[idx].to(self.device))
                value_loss = cross_loss(
                    values,
                    scalar_to_support(
                            self.buffer.td_values[idx].to(self.device),
                            self.value_support
                        )
                    )
                reward_loss = torch.zeros(1, device = self.device)

                old_idx = idx.clone()
                for i in range(self.buffer.K):
                    hidden_states = scale_gradient(hidden_states, 0.5)

                    actions_t = torch.zeros((len(old_idx),) + self.mcts.action_dim, device = self.device)
                    actions_t += (self.buffer.actions[old_idx].to(self.device) / count_of_actions).view(-1, 1, 1, 1)

                    hidden_states, rewards, policies, values =\
                        self.model.recurrent_inference(
                            hidden_states, actions_t
                        )

                    reward_loss += scale_gradient(
                        cross_loss(
                            rewards,
                            scalar_to_support(
                                self.buffer.rewards[old_idx].to(self.device),
                                self.reward_support
                            )
                        ),
                        self.buffer.K_ratio
                    )

                    next_idx, nonzero_idx = self.buffer.non_terminals_idx(old_idx)
                    hidden_states = hidden_states[nonzero_idx]

                    policy_loss += scale_gradient(
                        cross_loss(
                            policies[nonzero_idx],
                            self.buffer.policies[next_idx].to(self.device)),
                        self.buffer.K_ratio
                    )

                    value_loss += scale_gradient(
                        cross_loss(
                            values[nonzero_idx],
                            scalar_to_support(
                                self.buffer.td_values[next_idx].to(self.device),
                                self.value_support
                            )
                        ),
                        self.buffer.K_ratio
                    )

                    old_idx = next_idx

                loss = policy_loss + 0.25 * value_loss + reward_loss
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                s_policy += policy_loss.item()
                s_value += value_loss.item()
                s_reward += reward_loss.item()
            
            logs_loss += '\n' + str(iteration) + ',' \
                         + str(iteration * count_of_envs) + ',' \
                         + str(score.get_count_of_episodes()) + ',' \
                         + str(avg_score) + ',' \
                         + str(s_policy / self.update_target_iteration) + ',' \
                         + str(s_value / self.update_target_iteration) + ',' \
                         + str(s_reward / self.update_target_iteration)

            s_policy, s_value, s_reward = 0, 0, 0

            if iteration % self.update_target_iteration == 0 and iteration > 0:
                if iteration % self.update_t_iteration == 0 and iteration > self.update_t_iteration:
                    root = iteration // self.update_t_iteration - 1
                    t = min(2**root, self.min_t)
                    print('T has been changed from ', self.mcts.T, 'to', t)
                    self.mcts.T = t
                    
                write_to_file(logs_score, self.path + 'logs_score.txt')
                write_to_file(logs_loss, self.path + 'logs_loss.txt')

                with torch.no_grad():
                    self.buffer.update(self.model, self.mcts, self.value_support, self.reward_support)

                self.targ_model = deepcopy(self.model.value_model).to(self.device)

            torch.cuda.empty_cache()
