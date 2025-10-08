from dlgo.encoders.base import Encoder
from dlgo.gotypes import Point, Player
import numpy as np

"""
Feature name            num of planes   Description
Stone colour            3               Player stone / opponent stone / empty
Ones                    1               A constant plane filled with 1
"""

FEATURE_OFFSETS = {
    "stone_color": 0,
    "ones": 3,
    "current_player_color": 4
}


def offset(feature):
    return FEATURE_OFFSETS[feature]


class FourplaneEncoder(Encoder):
    def __init__(self, board_size=(19, 19), use_player_plane=True):
        self.board_width, self.board_height = board_size
        self.use_player_plane = use_player_plane
        self.num_planes = 4 + use_player_plane

    def name(self):
        return 'alphago'

    def encode(self, game_state):
        board_tensor = np.zeros((self.num_planes, self.board_height, self.board_width))
        for r in range(self.board_height):
            for c in range(self.board_width):
                point = Point(row=r + 1, col=c + 1)

                go_string = game_state.board.get_go_string(point)
                if go_string and go_string.color == game_state.next_player:
                    board_tensor[offset("stone_color")][r][c] = 1
                elif go_string and go_string.color == game_state.next_player.other:
                    board_tensor[offset("stone_color") + 1][r][c] = 1
                else:
                    board_tensor[offset("stone_color") + 2][r][c] = 1

                board_tensor[offset("ones")] = self.ones()
                if self.use_player_plane:
                    if game_state.next_player == Player.black:
                        board_tensor[offset("ones")] = self.ones()

        return board_tensor

    def ones(self):
        return np.ones((1, self.board_height, self.board_width))

    def zeros(self):
        return np.zeros((1, self.board_height, self.board_width))

    def encode_point(self, point):
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return self.num_planes, self.board_height, self.board_width


def create(board_size):
    return FourplaneEncoder(board_size)
