from dlgo.gosgf import Sgf_game
from dlgo.goboard import Board, GameState, Player, Point, Move
from dlgo.encoders.base import get_encoder_by_name
import numpy as np
import shutil
import tarfile
import gzip
import os


__all__ = [
    'DataProcessor'
]

class DataProcessor:
    def __init__(self, encoder, data_dir):
        self.encoder = get_encoder_by_name(encoder, 19)
        self.data_dir = data_dir
    
    def process_sgf_files(self, zip_file_name = None, file_list = None):
        board_size = 19
        zip_file = None
        if zip_file_name is not None:
            # remove gzip compression
            tar_file = self.unzip_data(zip_file_name)
            # open tarfile
            zip_file = tarfile.open(self.data_dir + '/' + tar_file)
            # get name of all files
            file_list = zip_file.getnames()
        # total number of moves
        print(f"zip_file: {zip_file_name}")
        total_examples = self.num_total_examples(zip_file, file_list)
        print(f"total examples: {total_examples}")
        shape = self.encoder.shape()
        feature_shape = np.insert(shape, 0, np.asarray([total_examples]))
        features = np.zeros(feature_shape)
        labels = np.zeros((total_examples,))
        
        counter = 0
        for name in file_list[1:]:
            # read sgf content as string
            if zip_file:
                with zip_file.extractfile(name) as file:
                    sgf_content = file.read()
            else:
                with open(name, 'r') as file:
                    sgf_content = file.read()
            # Now we need to parse the string into a readable python object
            # create sgf game from string. 
            # parses the string and creates a Sgf_Game object
            sgf = Sgf_game.from_string(sgf_content)
            board_size = sgf.get_size()
            go_board = Board(board_size, board_size)
            move = None
            game_state = GameState.new_game(board_size)

            # Now we get the sgf game object to replay each move
            # we play handicap moves first and get the game state
            game_state, first_move_done = self.get_handicap(sgf)
            
            # then we play the moves from the sequence
            # for every game state, we encode the game state and append to features
            # and next move we encode as label
            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    if move_tuple is not None:
                        row, col = move_tuple
                        # point has 1 based indexing
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
        base_name = zip_file_name.replace('.tar.gz', '') if zip_file_name else 'kgs-server-'
        data_file_name = self.data_dir + '/' + base_name
        train_feature_file_template = data_file_name + '_train_features'
        train_label_file_template = data_file_name + '_train_labels'
        test_feature_file_template = data_file_name + '_test_features'
        test_label_file_template = data_file_name + '_test_labels'

        indices = np.arange(len(features))
        np.random.shuffle(indices)
        split_idx = int(0.8 * len(features))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        X_train = features[train_indices]
        X_test = features[test_indices]
        y_train = labels[train_indices]
        y_test = labels[test_indices]

        np.save(train_feature_file_template, X_train)
        np.save(train_label_file_template, y_train)
        np.save(test_feature_file_template, X_test)
        np.save(test_label_file_template, y_test)

        return
        
    """Total number of moves each player plays throughout all
       the games provided.
    """
    def num_total_examples(self, zip_file, file_list):
        total_examples = 0
        for name in file_list:
            # read the file if the file is an sgf file
            if name.endswith('.sgf'):
                if zip_file:
                    with zip_file.extractfile(name) as file:
                        sgf_content = file.read()
                else:
                    with open(name, 'r') as file:
                        sgf_content = file.read()
                # get an sgf string from the file for further processing
                sgf = Sgf_game.from_string(sgf_content)

                # this is important because we need to start counting after the first move
                # if the game has handicap, then main sequence starts with 2nd move. otherwise,
                # main sequence starts with first move and we need to skip it
                game_state, first_move_done = self.get_handicap(sgf)

                num_moves = 0
                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None:
                        if first_move_done:
                            num_moves += 1
                        first_move_done = True
                total_examples = total_examples + num_moves
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
        board_size = sgf.get_size()
        go_board = Board(board_size, board_size)
        first_move_done = False
        move = None
        game_state = GameState.new_game(board_size)
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    go_board.place_stone(Player.black, Point(row + 1, col + 1))
            first_move_done = True
            game_state = GameState(go_board, Player.white, None, move)
        return game_state, first_move_done
    
    def combine_numpy_files(self, base_name, file_type, output_filename=None, matching_files = None):
        """
        Generic function to combine multiple numpy files into a single file.
        
        Args:
            data_dir (str): Directory containing the numpy files
            base_name (str): Base name to match files (e.g., 'KGS-2019_04-19-1255-_train')
            file_type (str): Type of files to combine ('features' or 'labels')
            output_filename (str, optional): Output filename. If None, auto-generates one.
        
        Returns:
            str: Path to the combined output file
        """
        
        # Sort files to ensure consistent ordering
        matching_files.sort()
        print(f"Found {len(matching_files)} {file_type} files to combine:")
        # Load and combine all files
        combined_data = []
        for file in matching_files:
            file_path = os.path.join(self.data_dir, file)
            data = np.load(file_path)
            combined_data.append(data)
            print(f"  Loaded {file}: shape {data.shape}")
        
        # Concatenate along the first axis (assuming these are training samples)
        final_data = np.concatenate(combined_data, axis=0)
        print(f"Combined shape: {final_data.shape}")
        
        # Generate output filename if not provided
        if output_filename is None:
            output_filename = f"{base_name}_{file_type}_combined.npy"
        
        output_path = os.path.join(self.data_dir, output_filename)
        
        # Save combined data
        np.save(output_path, final_data)
        print(f"Combined {file_type} saved to: {output_filename}")
        
        return output_path       

