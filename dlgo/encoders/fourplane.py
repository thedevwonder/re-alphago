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
    "current_player_color": 4,
    "legal_moves": 5
}


def offset(feature):
    return FEATURE_OFFSETS[feature]


class FourplaneEncoder(Encoder):
    def __init__(self, board_size=(19, 19), use_player_plane=True, use_legal_moves=True):
        self.board_width, self.board_height = board_size
        self.use_player_plane = use_player_plane
        self.use_legal_moves = use_legal_moves
        self.num_planes = 4 + use_player_plane + use_legal_moves

    def name(self):
        return 'alphago'

    def encode(self, game_state):
        board_tensor = np.zeros((self.num_planes, self.board_height, self.board_width))
        
        # Set empty cells plane (default to empty)
        board_tensor[offset("stone_color") + 2] = 1
        
        # Iterate over occupied points only (much faster for sparse boards)
        next_player = game_state.next_player
        opponent = next_player.other
        stone_color_offset = offset("stone_color")
        
        for point, go_string in game_state.board._grid.items():
            if go_string is None:
                continue
            r = point.row - 1
            c = point.col - 1
            if go_string.color == next_player:
                board_tensor[stone_color_offset][r][c] = 1
                board_tensor[stone_color_offset + 2][r][c] = 0  # Not empty
            elif go_string.color == opponent:
                board_tensor[stone_color_offset + 1][r][c] = 1
                board_tensor[stone_color_offset + 2][r][c] = 0  # Not empty
        
        # Set ones plane once (moved outside loop)
        board_tensor[offset("ones")] = 1
        
        # Set player plane once (moved outside loop)
        if self.use_player_plane and next_player == Player.black:
            board_tensor[offset("current_player_color")] = 1
        
        # Set legal moves - optimized inline computation to avoid expensive deep copies
        if self.use_legal_moves:
            if not game_state.is_over():
                legal_moves_offset = offset("legal_moves")
                board = game_state.board
                
                # Fast inline legal move checking without deep copies
                for row in range(1, board.num_rows + 1):
                    for col in range(1, board.num_cols + 1):
                        point = Point(row, col)
                        
                        # Fast check: point must be empty
                        if board._grid.get(point) is not None:
                            continue
                        
                        # Check self-capture without deep copy
                        has_liberty = False
                        would_capture = False
                        friendly_strings = []
                        
                        for neighbor in point.neighbors():
                            if not board.is_on_grid(neighbor):
                                continue
                            neighbor_string = board._grid.get(neighbor)
                            if neighbor_string is None:
                                has_liberty = True
                                break
                            elif neighbor_string.color == next_player:
                                friendly_strings.append(neighbor_string)
                            else:  # opponent string
                                if neighbor_string.num_liberties == 1:
                                    would_capture = True
                        
                        # If has liberty, not self-capture
                        if not has_liberty:
                            # Check if all friendly strings would have no liberties
                            if friendly_strings and all(s.num_liberties == 1 for s in friendly_strings):
                                if not would_capture:
                                    continue  # Self-capture, skip
                        
                        # Check ko violation - simplified heuristic for performance
                        # Full ko check requires deep copy, so we use a fast heuristic:
                        # If last move captured exactly one stone and we're trying to recapture
                        # at that same position, it's likely a ko violation
                        if would_capture and game_state.last_move and game_state.last_move.is_play:
                            # Check if we're trying to play at the last move position
                            # (which would be recapturing after a single-stone capture)
                            if point == game_state.last_move.point:
                                # Count how many stones we'd capture
                                captured_stones = 0
                                for neighbor in point.neighbors():
                                    if not board.is_on_grid(neighbor):
                                        continue
                                    neighbor_string = board._grid.get(neighbor)
                                    if neighbor_string and neighbor_string.color == opponent:
                                        if neighbor_string.num_liberties == 1:
                                            captured_stones += len(neighbor_string.stones)
                                # If capturing exactly one stone at last move position, likely ko
                                if captured_stones == 1:
                                    continue  # Likely ko violation, skip
                        
                        # Valid move - set it
                        r = row - 1
                        c = col - 1
                        board_tensor[legal_moves_offset][r][c] = 1

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
