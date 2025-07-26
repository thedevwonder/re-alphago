from dlgo import agent, mcts
from dlgo import goboard as goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move
import time
import uuid
import os


def main():
    
    board_size = 9
    game = goboard.GameState.new_game(board_size)
    bots = {
        gotypes.Player.black: mcts.MCTSAgent(temperature=1.41, num_rounds=50),
        gotypes.Player.white: agent.FastRandomBot(),
    }
    # Generate random game id and SGF file
    game_id = str(uuid.uuid4())
    sgf_path = os.path.join('games', f'mcts_v_b_{game_id}.sgf')
    with open(sgf_path, 'w') as sgf_file:
        sgf_file.write(f"(;GM[1]FF[4]SZ[{board_size}]\n")
        while not game.is_over():
            time.sleep(0.3)
            print(chr(27) + "[2J")
            print_board(game.board)
            bot_move = bots[game.next_player].select_move(game)
            print_move(game.next_player, bot_move, sgf_file=sgf_file)
            game = game.apply_move(bot_move)
        winner = game.winner()
        if winner:
            sgf_file.write(f"RE[{winner.name}+R]\n")
        sgf_file.write(")\n")
    print(f"winner: {winner}")
    print(f"SGF saved to: {sgf_path}")


if __name__ == '__main__':
    main()