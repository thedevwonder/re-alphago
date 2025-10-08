import torch.nn as nn
import torch
import random
from sl_policy_network import SLPolicyBot, SLPolicyNetwork
import torch.optim as optim
from dlgo.goboard import GameState

class RLPolicyTrainer:
    # we need to modify the hyperparameters
    def __init__(self):
        self.model = torch.load('sl_policy_network.pth')
        self.opponents = []
        self.batch_size = 128
        self.optimizer = optim.SGD(self.model.parameters(), 0.003)
        self.board_size = 19
        self.num_of_episodes = 100000

    def self_play(self, game_state):
        trajectory = []
        opponent_model = self.model
        # choose a random model as your opponent
        if len(self.opponents) > 0:
            opponent_model = random.choices(self.opponents)
        your_agent = SLPolicyBot(self.model)
        opponent_agent = SLPolicyBot(opponent_model)
        while not game_state.is_over():
            if game_state.next_player == 0:
                action, log_prob = your_agent.select_move(game_state)
                # we also need to decode this action and convert it to type Move
                trajectory.append((game_state, action, log_prob))
            else:
                action = opponent_agent.select_move(game_state)
            game_state = game_state.apply_move(action)
        
        winner = game_state.winner()
        outcome = 1 if winner == 0 else -1
        return outcome, trajectory

    def update_policy(self, trajectories_outcomes):
        total_loss = 0
        for outcome, trajectory in trajectories_outcomes:
            loss = 0
            for state, action, log_prob in trajectory:
                loss += outcome * log_prob
            total_loss += loss

        #normalize the total_loss
        total_loss = total_loss / len(trajectories_outcomes)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def train_batch(self):
        trajectories_outcomes = []
        for i in range(1, self.batch_size+1):
            game_init_state = GameState.new_game(self.board_size)
            outcome, trajectory = self.self_play(game_init_state)
            trajectories_outcomes.append((outcome, trajectory))
        
        self.update_policy(trajectories_outcomes=trajectories_outcomes)

    def train(self):
        for i in range(1, self.num_of_episodes+1):
            self.train_batch()
            if i % 500 and i > 0:
                opponent_model = SLPolicyNetwork()
                opponent_model.load_state_dict(self.model.state_dict())
                self.opponents.append(opponent_model)
            