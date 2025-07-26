from dlgo import gotypes
import numpy as np

COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: ' . ',
    gotypes.Player.black: ' x ',
    gotypes.Player.white: ' o ',
}

def point_to_sgf_coords(point):
    # SGF coords: 'a' for 1, 'b' for 2, ...
    return chr(ord('a') + point.col - 1) + chr(ord('a') + point.row - 1)


def print_move(player, move, return_sgf=False, sgf_file=None):
    if move.is_pass:
        move_str = 'passes'
        sgf_str = f';{"B" if player == gotypes.Player.black else "W"}[]'
    elif move.is_resign:
        move_str = 'resigns'
        sgf_str = f'C[{player} resigns]'
    else:
        move_str = '%s%d' % (COLS[move.point.col - 1], move.point.row)
        color = 'B' if player == gotypes.Player.black else 'W'
        coords = point_to_sgf_coords(move.point)
        sgf_str = f';{color}[{coords}]'
    print('%s %s' % (player, move_str))
    if sgf_file is not None and sgf_str:
        print(sgf_str)
        sgf_file.write(sgf_str + "\n")


def print_board(board):
    for row in range(board.num_rows, 0, -1):
        bump = " " if row <= 9 else ""
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(gotypes.Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%s%d %s' % (bump, row, ''.join(line)))
    print('    ' + '  '.join(COLS[:board.num_cols]))



# this feature is only used in goboard_fast.py
class MoveAge():
    def __init__(self, board):
        self.move_ages = - np.ones((board.num_rows, board.num_cols))

    def get(self, row, col):
        return self.move_ages[row, col]

    def reset_age(self, point):
        self.move_ages[point.row - 1, point.col - 1] = -1

    def add(self, point):
        self.move_ages[point.row - 1, point.col - 1] = 0

    def increment_all(self):
        self.move_ages[self.move_ages > -1] += 1