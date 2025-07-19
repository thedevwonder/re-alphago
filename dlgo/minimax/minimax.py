import enum
import random
from dlgo.goboard import Move

from dlgo.agent import Agent

__all__ = [
    'MinimaxAgent',
]

class GameResult(enum.Enum):
    loss = 1
    draw = 2
    win = 3

def reverse_game_result(game_result):
    if game_result.win:
        return GameResult.loss
    elif game_result.loss:
        return GameResult.win
    else:
        return GameResult.draw
    

def best_result(game_state):
    # I want to fetch all the legal moves for a game state
    # this gives me the breadth,
    # for every move I wanna calculate the opponent's best result
    # if the best result is lose, I wanna select that because I want opp to lose
    if game_state.is_over():
        if game_state.winner() == game_state.next_player:
            return GameResult.win
        if game_state.winner() is None:
            return GameResult.draw
        return GameResult.loss    
    
    best_result_so_far = GameResult.loss
    for move in game_state.legal_moves():
        next_game_state = game_state.apply_move(move)
        opponent_best_result = best_result(next_game_state)
        our_best_result = reverse_game_result(opponent_best_result)
        if our_best_result.value > best_result_so_far.value:
            best_result_so_far = our_best_result
    return best_result_so_far

    
class MinimaxAgent(Agent):
    def select_move(self, game_state):
        winning_moves = []
        losing_moves = []
        draw_moves = []

        # we are selecting a move, we definitely wanna select a winning move first
        # we are gonna select drawing move if not a winning move,
        # and at the end we are gonna select a losing move if nothing else comes to mind,
        # usually players kind of resign if they see a losing game -> so better resign
        for move in game_state.legal_moves():
            next_game_state = game_state.apply_move(move)
            opponent_best_result = best_result(next_game_state)
            our_best_result = reverse_game_result(opponent_best_result)
            if our_best_result == GameResult.win:
                winning_moves.append(our_best_result)
            elif our_best_result == GameResult.draw:
                draw_moves.append(our_best_result)
            else:
                losing_moves.append(our_best_result)
        if winning_moves:
            return random.choice(winning_moves)
        elif draw_moves:
            return random.choice(draw_moves)
        else:
            # opponent will always not chose the best move 
            # so even if you have a losing move you play instead of resigning
            # return Move.resign()
            return random.choice(losing_moves)

