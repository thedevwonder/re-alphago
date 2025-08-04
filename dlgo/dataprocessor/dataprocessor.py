from dlgo.gosgf import Sgf_game
from dlgo.goboard import Board, GameState, Player, Point, Move
from dlgo.encoders.base import get_encoder_by_name
import numpy as np
import shutil
import tarfile
import gzip


class DataProcessor:
    def __init__(self, encoder, data_dir):
        self.encoder = get_encoder_by_name(encoder, 19)
        self.data_dir = data_dir
    
    def process_sgf_files(self, zip_file_name):
        # remove gzip compression
        tar_file = self.unzip_data(zip_file_name)
        # open tarfile
        zip_file = tarfile.open(self.data_dir + '/' + tar_file)
        # get name of all files
        file_list = zip_file.getnames()
        no_of_files = len(file_list)
        # divide training and test 80/20
        game_list = range(int(0.8 * no_of_files))
        total_examples = self.num_total_examples(zip_file, file_list, game_list)
        shape = self.encoder.shape()
        feature_shape = np.insert(shape, 0, np.asarray([total_examples]))
        features = np.zeros(feature_shape)
        labels = np.zeros((total_examples,))
        
        counter = 0
        for index in game_list:
            name = file_list[index+1]
            # read sgf content as string
            sgf_content = zip_file.extractfile(name).read()

            # Now we need to parse the string into a readable python object
            # create sgf game from string. 
            # parses the string and creates a Sgf_Game object
            sgf = Sgf_game.from_string(sgf_content)
            go_board = Board(19, 19)
            move = None
            game_state = GameState.new_game(19)

            # Now we get the sgf game object to replay each move

            game_state, first_move_done = self.get_handicap(sgf)
            # first we place handicap stones to give weaker (black) some advantage
            if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
                for setup in sgf.get_root().get_setup_stones():
                    for move in setup:
                        row, col = move
                        go_board.place_stone(Player.black, Point(row + 1, col + 1))  # black gets handicap
                first_move_done = True
                game_state = GameState(go_board, Player.white, None, move)
            
            # then we play the moves from the sequence
            # for every game state, we encode the game state and append to features
            # and next move we encode as label
            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    if move_tuple is not None:
                        row, col = move_tuple
                        point = Point(row + 1, col + 1)
                        move = Move.play(point)
                    else:
                        move = Move.pass_turn()
                    if first_move_done and point is not None:
                        features[counter] = self.encoder.encode(game_state)
                        labels[counter] = self.encoder.encode_point(point)
                        counter += 1
                    game_state = game_state.apply_move(move)
                    first_move_done = True

        # Save the processed data
        base_name = zip_file_name.replace('.tar.gz', '')
        data_file_name = self.data_dir + '/' + base_name + '_train'
        feature_file_template = data_file_name + '_features_%d'
        label_file_template = data_file_name + '_labels_%d'
        chunk = 0
        chunksize = 1024
        # divide if the shape is > 1024
        while features.shape[0] >= chunksize:
            feature_file = feature_file_template % chunk
            label_file = label_file_template % chunk
            chunk += 1
            current_features, features = features[:chunksize], features[chunksize:]
            current_labels, labels = labels[:chunksize], labels[chunksize:]
            np.save(feature_file, current_features)
            np.save(label_file, current_labels)
        if features.shape[0] > 0:
            feature_file = feature_file_template % chunk
            label_file = label_file_template % chunk
            np.save(feature_file, features)
            np.save(label_file, labels)
        
        return
    
    def num_total_examples(self, zip_file, file_list, game_list):
        total_examples = 0
        for index in game_list:
            name = file_list[index + 1]
            if name.endswith('.sgf'):
                sgf_content = zip_file.extractfile(name).read()
                sgf = Sgf_game.from_string(sgf_content)
                game_state, first_move_done = self.get_handicap(sgf)

                num_moves = 0
                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None:
                        if first_move_done:
                            num_moves += 1
                        first_move_done = True
                total_examples = total_examples + num_moves
            else:
                raise ValueError(name + ' is not a valid sgf')
        return total_examples
    
    def unzip_data(self, zip_file_name):
        this_gz = gzip.open(self.data_dir + '/' + zip_file_name)  # <1>

        tar_file = zip_file_name[0:-3]  # <2>
        this_tar = open(self.data_dir + '/' + tar_file, 'wb')

        shutil.copyfileobj(this_gz, this_tar)  # <3>
        this_tar.close()
        return tar_file
    
    @staticmethod
    def get_handicap(sgf):
        go_board = Board(19, 19)
        first_move_done = False
        move = None
        game_state = GameState.new_game(19)
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    go_board.place_stone(Player.black, Point(row + 1, col + 1))
            first_move_done = True
            game_state = GameState(go_board, Player.white, None, move)
        return game_state, first_move_done

