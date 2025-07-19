import random

from dlgo.agent import Agent
from dlgo.gotypes import Player

__all__ = [
    'AlphaBetaAgent',
]

MAX = 999999
MIN = -999999


def alpha_beta_result(game_state, max_depth, best_black, best_white, eval_fn):
    if game_state.is_over():
        if game_state.winner() == game_state.next_player:
            return MAX
        if game_state.winner() is None:
            return eval_fn(game_state)
        return MIN
    
    if max_depth == 0:
        return eval_fn(game_state)

    best_result_so_far = MIN
    for move in game_state.legal_moves():
        next_game_state = game_state.apply_move(move)
        opponent_best_result = alpha_beta_result(next_game_state, max_depth=max_depth-1, eval_fn=eval_fn)
        our_best_result = -1*opponent_best_result
        if our_best_result > best_result_so_far:
            best_result_so_far = our_best_result

        '''
            Here's the GPT explaination of how the pruning code works

            Suppose we're searching a game tree for the best move for White.
            best_black = 5 (Black has already found a move that guarantees at least 5 for themselves)
            We're evaluating a move for White, and after some calculation, best_result_so_far = 2 (White can only guarantee 2 here)
            outcome_for_black = -1 * 2 = -2
            Now, check:
            if outcome_for_black < best_black:
            if -2 < 5: → True
            This means:
            Black can already guarantee a better result (5) elsewhere.
            So, there's no point in White continuing to search this branch, because Black will never let the game reach this position.
            We prune this branch.
        '''
        if game_state.next_player == Player.white:
            if best_result_so_far > best_white:                       # <8>
                best_white = best_result_so_far                       # <8>
            outcome_for_black = -1 * best_result_so_far               # <9>
            if outcome_for_black < best_black:                 # <9>
                return best_result_so_far                             # <9>
# end::alpha-beta-prune-2[]
# tag::alpha-beta-prune-3[]
        elif game_state.next_player == Player.black:
            if best_result_so_far > best_black:                       # <10>
                best_black = best_result_so_far                       # <10>
            outcome_for_white = -1 * best_result_so_far               # <11>
            if outcome_for_white < best_white:                 # <11>
                return best_result_so_far 
            
    return best_result_so_far
    

# tag::alpha-beta-agent[]
class AlphaBetaAgent(Agent):
    def __init__(self, max_depth, eval_fn):
        Agent.__init__(self)
        self.max_depth = max_depth
        self.eval_fn = eval_fn

    def select_move(self, game_state):
        # keep a best moves list
        best_moves = []
        # keep a best global scorre, a best white and a best black score
        best_score = None
        best_white_score = MIN
        best_black_score = MIN

        # we calculate our best result
        # we still need to maintain best moves and select random best move
        for move in game_state.legal_moves():
            our_move = game_state.apply_move(move)
            opponent_best_result = alpha_beta_result(our_move, self.max_depth, best_black_score, best_white_score, self.eval_fn)
            our_best_result = -1*opponent_best_result
            if (not best_moves) or our_best_result > best_score:
                best_moves = [move]
                best_score = our_best_result
                if game_state.next_player == Player.white:
                    best_white_score = our_best_result

                if game_state.next_player == Player.black:
                    best_black_score = our_best_result
            elif our_best_result == best_score:
                best_moves.append(move)
        return random.choice(best_moves)