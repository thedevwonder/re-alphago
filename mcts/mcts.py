import math
import random

from dlgo import agent
from dlgo.gotypes import Player

__all__ = [
    'MCTSAgent',
]   


class MCTSNode(object):
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {
            Player.black: 0,
            Player.white: 0,
        }
        self.num_rollouts = 0
        self.children = []
        self.unvisited_moves = game_state.legal_moves()

    def add_random_child(self):
        # select a child at random
        # pop it out of unvisited moves
        # apply move to get child game state
        # create a child MCTS node
        # add it to the children list of current node

        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        child_game_state = self.game_state.apply_move(new_move)
        child_node = MCTSNode(child_game_state, self, new_move)
        self.children.append(child_node)
        return child_node        

    def record_win(self, winner):
        # we already know who the winner is from func arg
        # at the end of the game, we will increment
        # num_of_rollout: total simulated games
        # win_counts: which track black and white wins per node
        self.num_rollouts += 1
        self.win_counts[winner] += 1
        
    def can_add_child(self):
        return len(self.unvisited_moves) > 0
        

    def is_terminal(self):
        return self.game_state.is_over()

    def winning_frac(self, player):
        return float(self.win_counts[player]) / float(self.num_rollouts)


class MCTSAgent(agent.Agent):
    def __init__(self, num_rounds, temperature):
        agent.Agent.__init__(self)
        # num of simulations
        self.num_rounds = num_rounds
        # temperature
        self.temperature = temperature


    def select_move(self, game_state):
        # first we need to simulate games
        # once we simulate games, we collect scores and develop a statistics
        # once we develop scores for all the children, we select the child with the best score

        # initialize the root node
        root = MCTSNode(game_state)        # for num_of_rounds, run the loop
        for i in range(self.num_rounds):
            node = root
            # selection
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            # We select a random child out of unvisited children. selection
            if node.can_add_child():
                node = node.add_random_child()

            # Simulate a random game from this node. Rollout
            winner = self.simulate_random_game(node.game_state)

            # Propagate scores back up the tree. Backpropogation
            while node is not None:
                node.record_win(winner)
                node = node.parent

        # Having performed as many MCTS rounds as we have time for, we
        # now pick a move.
        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_frac(game_state.next_player)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        print('Select move %s with win pct %.3f' % (best_move, best_pct))
        return best_move
            
    def select_child(self, node):
        """Select a child according to the upper confidence bound for
        trees (UCT) metric.
        """
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = math.log(total_rollouts)

        best_score = -1
        best_child = None
        # Loop over each child.
        for child in node.children:
            # Calculate the UCT score.
            win_percentage = child.winning_frac(node.game_state.next_player)
            exploration_factor = math.sqrt(log_rollouts / child.num_rollouts)
            uct_score = win_percentage + self.temperature * exploration_factor
            # Check if this is the largest we've seen so far.
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child
    @staticmethod
    def simulate_random_game(game):
        bots = {
            Player.black: agent.FastRandomBot(),
            Player.white: agent.FastRandomBot(),
        }
        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        return game.winner()