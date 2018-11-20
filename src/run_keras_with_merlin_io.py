################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://github.com/CSTR-Edinburgh/merlin
#
#                Centre for Speech Technology Research
#                     University of Edinburgh, UK
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.
#
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute
#  this software and its documentation without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of this work, and to
#  permit persons to whom this work is furnished to do so, subject to
#  the following conditions:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   - The authors' names may not be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
################################################################################

from builtins import object
import os
import sys
import time
import numpy as np

from keras_lib import configuration
from keras_lib import data_utils
from keras_lib.train import TrainKerasModels

from frontend.label_normalisation import HTSLabelNormalisation
from frontend.silence_remover import SilenceRemover
from frontend.acoustic_composition import AcousticComposition


class KerasClass(object):

    def __init__(self, cfg):

        # model type (duration or acoustic)
        self.model_output_type = cfg.model_output_type

        # ----------------------------------------------------
        # ------------------- Input-Output -------------------
        # ----------------------------------------------------

        self.label_type = cfg.label_type
        self.cmp_ext = cfg.cmp_ext
        inp_file_ext = cfg.inp_file_ext
        out_file_ext = cfg.out_file_ext
        self.label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name,
                                                      add_frame_features=cfg.add_frame_features == 'True',  # must be bool
                                                      subphone_feats=cfg.subphone_feats)

        # Create streams files (they store data from dimension dictionaries for synthesis)
        in_streams = sorted(cfg.in_dimension_dict.keys())
        indims = [str(cfg.in_dimension_dict[s]) for s in in_streams]
        self.out_streams = sorted(cfg.out_dimension_dict.keys())
        self.outdims = [str(cfg.out_dimension_dict[s]) for s in self.out_streams]

        with open(os.path.join(cfg.model_dir, 'stream_info.txt'), 'w') as f:
            f.write(' '.join(in_streams) + '\n')
            f.write(' '.join(indims) + '\n')
            f.write(' '.join(self.out_streams) + '\n')
            f.write(' '.join(self.outdims) + '\n')

        # Input output dimensions
        self.inp_dim = cfg.inp_dim
        if self.model_output_type == 'duration':
            self.out_dim = cfg.dur_dim
        elif self.model_output_type == 'acoustic':
            self.out_dim = cfg.cmp_dim

        # Data normalization method
        self.inp_norm = cfg.inp_norm
        self.out_norm = cfg.out_norm

        # Norm stats files
        self.inp_stats_file = cfg.inp_stats_file
        self.out_stats_file = cfg.out_stats_file

        self.inp_scaler = None
        self.out_scaler = None

        # ---------------------------------------------------
        # ------------------- Directories -------------------
        # ---------------------------------------------------

        self.plot_dir = os.path.join(cfg.plot_dir, cfg.nnets_file_name)
        # Select data directories based on model input-output type
        if self.model_output_type == 'duration':

            # Input
            self.inp_feat_dir = cfg.inp_feat_dir_dur
            self.bin_lab_dir = cfg.bin_lab_dir_dur
            self.bin_lab_dir_nosilence = cfg.bin_lab_dir_dur_nosilence
            self.bin_lab_dir_nosilence_norm = cfg.bin_lab_dir_dur_nosilence_norm

            # Output
            self.out_feat_dir = cfg.out_feat_dir_dur
            self.out_feat_dir_norm = cfg.out_feat_dir_dur_norm

        elif self.model_output_type == 'acoustic':

            # Input
            self.inp_feat_dir = cfg.inp_feat_dir_cmp
            self.bin_lab_dir = cfg.bin_lab_dir_cmp
            self.bin_lab_dir_nosilence = cfg.bin_lab_dir_cmp_nosilence
            self.bin_lab_dir_nosilence_norm = cfg.bin_lab_dir_cmp_nosilence_norm

            # Output
            # self.out_feat_dir = cfg.out_feat_dir_cmp
            # self.out_feat_dir_norm = cfg.out_feat_dir_cmp_norm
            self.out_feat_dir = cfg.nn_cmp_dir
            self.out_feat_dir_norm = cfg.nn_cmp_norm_dir

            # self.nn_cmp_dir = cfg.nn_cmp_dir
            # self.nn_cmp_norm_dir = cfg.nn_cmp_norm_dir

        else:
            print("invalid model output type")
            raise

        # --------------------------------------------------------
        # ------------------- Model Parameters -------------------
        # --------------------------------------------------------

        self.hidden_layer_type = cfg.hidden_layer_type
        self.hidden_layer_size = cfg.hidden_layer_size

        self.sequential_training = cfg.sequential_training

        self.stateful = cfg.stateful
        self.batch_size = cfg.batch_size
        self.seq_length = cfg.seq_length

        self.training_algo = cfg.training_algo
        self.shuffle_data = cfg.shuffle_data

        self.output_layer_type = cfg.output_layer_type
        self.loss_function = cfg.loss_function
        self.optimizer = cfg.optimizer

        self.rnn_params = cfg.rnn_params
        self.dropout_rate = cfg.dropout_rate
        self.num_of_epochs = cfg.num_of_epochs

        self.json_model_file = cfg.json_model_file
        self.h5_model_file = cfg.h5_model_file

        # -----------------------------------------------------------
        # ------------------- Generate file lists -------------------
        # -----------------------------------------------------------

        train_file_number = cfg.train_file_number
        valid_file_number = cfg.valid_file_number
        test_file_number = cfg.test_file_number

        # List of file ids
        self.file_id_scp = cfg.file_id_scp

        # Create train, valid and test file lists
        self.file_id_list = data_utils.read_file_list(self.file_id_scp)
        self.train_id_list = self.file_id_list[0: train_file_number]
        self.valid_id_list = self.file_id_list[train_file_number: train_file_number + valid_file_number]
        self.test_id_list = self.file_id_list[train_file_number + valid_file_number: train_file_number + valid_file_number + test_file_number]

        # TODO: should the binary labels be split into training test validate as well?  These files only pertain to labels/input, the output data is already binary
        self.inp_feat_file_list = data_utils.prepare_file_path_list(self.file_id_list, self.inp_feat_dir, inp_file_ext)
        self.bin_lab_file_list = data_utils.prepare_file_path_list(self.file_id_list, self.bin_lab_dir, inp_file_ext)
        self.bin_lab_nosilence_file_list = data_utils.prepare_file_path_list(self.file_id_list, self.bin_lab_dir_nosilence, inp_file_ext)

        # Train, test, validation file lists
        self.inp_train_file_list = data_utils.prepare_file_path_list(self.train_id_list, self.bin_lab_dir_nosilence, inp_file_ext)
        self.out_train_file_list = data_utils.prepare_file_path_list(self.train_id_list, self.out_feat_dir, out_file_ext)
        self.inp_valid_file_list = data_utils.prepare_file_path_list(self.valid_id_list, self.bin_lab_dir_nosilence, inp_file_ext)
        self.out_valid_file_list = data_utils.prepare_file_path_list(self.valid_id_list, self.out_feat_dir, out_file_ext)
        self.inp_test_file_list = data_utils.prepare_file_path_list(self.test_id_list, self.bin_lab_dir_nosilence, inp_file_ext)
        self.out_test_file_list = data_utils.prepare_file_path_list(self.test_id_list, self.out_feat_dir, out_file_ext)

        # For cmp files generated as targets (applies to acoustic model only)
        self.nn_cmp_file_list = []
        self.nn_cmp_norm_file_list = []

        self.in_file_list_dict = {}
        for feature_name in list(cfg.in_dir_dict.keys()):
            self.in_file_list_dict[feature_name] = data_utils.prepare_file_path_list(self.file_id_list,
                                                                                     cfg.in_dir_dict[feature_name],
                                                                                     cfg.file_extension_dict[feature_name],
                                                                                     False)

        # self.gen_test_file_list = data_utils.prepare_file_path_list(self.test_id_list, pred_feat_dir, out_file_ext)

        # if self.GenTestList:
        #     test_id_list = data_utils.read_file_list(test_id_scp)
        #     self.inp_test_file_list = data_utils.prepare_file_path_list(test_id_list, inp_feat_dir, inp_file_ext)
        #     self.gen_test_file_list = data_utils.prepare_file_path_list(test_id_list, pred_feat_dir, out_file_ext)

        # ------------------------------------------------------
        # ------------------- Main Processes -------------------
        # ------------------------------------------------------

        self.MAKELAB = cfg.MAKELAB  # make binary labels (required step before normalization and training)
        self.MAKECMP = cfg.MAKECMP
        self.NORMDATA = cfg.NORMDATA  # normalizes input and output data, creates data scaling objects
        self.TRAINDNN = cfg.TRAINDNN  # train the Keras model
        self.TESTDNN = cfg.TESTDNN  # test the Keras model

        # ----------------------------------------------------------
        # ------------------- Define Keras Model -------------------
        # ----------------------------------------------------------

        self.keras_models = TrainKerasModels(self.inp_dim, self.hidden_layer_size, self.out_dim, self.hidden_layer_type,
                                             output_type=self.output_layer_type, dropout_rate=self.dropout_rate,
                                             loss_function=self.loss_function, optimizer=self.optimizer,
                                             rnn_params=self.rnn_params)

    def make_labels(self):

        # simple HTS labels
        print('preparing label data (input) using standard HTS style labels')

        if not os.path.isfile(self.bin_lab_file_list[-1]):
            # This does not normalize the data as the name suggests, rather translates it to binary
            self.label_normaliser.perform_normalisation(self.inp_feat_file_list, self.bin_lab_file_list,
                                                        label_type=self.label_type)

        # TODO: Additional features may be added in the future... parts of speech?  Some context for intonation?
        # if cfg.additional_features:
        #     out_feat_dir = os.path.join(cfg.data_dir, 'binary_label_%s_%s' % (cfg.label_type, str(self.inp_dim)))
        #     out_feat_file_list = data_utils.prepare_file_path_list(file_id_list, out_feat_dir, cfg.lab_ext)
        #     in_dim = self.label_normaliser.dimension
        #     for new_feature, new_feature_dim in cfg.additional_features.items():
        #         new_feat_dir = os.path.join(cfg.data_dir, new_feature)
        #         new_feat_file_list = data_utils.prepare_file_path_list(file_id_list, new_feat_dir, '.' + new_feature)
        #
        #         merger = MergeFeat(lab_dim=in_dim, feat_dim=new_feature_dim)
        #         merger.merge_data(binary_label_file_list, new_feat_file_list, out_feat_file_list)
        #         in_dim += new_feature_dim
        #
        #         binary_label_file_list = out_feat_file_list

        # This silence remover has little to no effect, no change in file 1
        if not os.path.isfile(self.bin_lab_nosilence_file_list[-1]):
            remover = SilenceRemover(n_cmp=self.inp_dim, silence_pattern=cfg.silence_pattern, label_type=cfg.label_type,
                                     remove_frame_features=cfg.add_frame_features, subphone_feats=cfg.subphone_feats)
            remover.remove_silence(self.bin_lab_file_list, self.inp_feat_file_list, self.bin_lab_nosilence_file_list)

    def make_cmp(self):

        # File lists for the final cmp files (these are re-generated below to fit a precise numpy data array)
        self.nn_cmp_file_list = data_utils.prepare_file_path_list(self.file_id_list, self.out_feat_dir, self.cmp_ext)
        self.nn_cmp_norm_file_list = data_utils.prepare_file_path_list(self.file_id_list, self.out_feat_dir_norm,
                                                                       self.cmp_ext)
        # TODO: Get the delta and acceleration windows from the recipe file.
        acoustic_worker = AcousticComposition(delta_win=[-0.5, 0.0, 0.5], acc_win=[1.0, -2.0, 1.0])

        # TODO: Lets try this at some point
        # if 'dur' in list(cfg.in_dir_dict.keys()) and cfg.AcousticModel:
        #     acoustic_worker.make_equal_frames(dur_file_list, lf0_file_list, cfg.in_dimension_dict)
        acoustic_worker.prepare_nn_data(self.in_file_list_dict, self.nn_cmp_file_list,
                                        cfg.in_dimension_dict, cfg.out_dimension_dict)

        remover = SilenceRemover(n_cmp=cfg.cmp_dim, silence_pattern=cfg.silence_pattern, label_type=cfg.label_type,
                                 remove_frame_features=cfg.add_frame_features, subphone_feats=cfg.subphone_feats)
        remover.remove_silence(self.nn_cmp_file_list[0:cfg.train_file_number + cfg.valid_file_number],
                               self.inp_feat_file_list[0:cfg.train_file_number + cfg.valid_file_number],
                               self.nn_cmp_file_list[0:cfg.train_file_number + cfg.valid_file_number])  # save to itself

    def normalize_data(self):

        # What type of normalization? -- its given as "method" in compute_norm_stats

        # Check if normalization stat files already exist
        if os.path.isfile(self.inp_stats_file) and os.path.isfile(self.out_stats_file):
            self.inp_scaler = data_utils.load_norm_stats(self.inp_stats_file, self.inp_dim, method=self.inp_norm)
            self.out_scaler = data_utils.load_norm_stats(self.out_stats_file, self.out_dim, method=self.out_norm)

        else:  # Create the scaler objects
            print('preparing train_x, train_y from input and output feature files...')
            train_x, train_y, train_flen = data_utils.read_data_from_file_list(self.inp_train_file_list, self.out_train_file_list,
                                                                            self.inp_dim, self.out_dim, sequential_training=self.sequential_training)

            print('computing norm stats for train_x...')
            # I have removed scaling from binary variables (discrete_dict columns are all binary)
            ind = [int(i) for i in self.label_normaliser.discrete_dict.keys()]
            self.inp_scaler = data_utils.compute_norm_stats(train_x,
                                                            self.inp_stats_file,
                                                            method=self.inp_norm,
                                                            no_scaling_ind=ind)

            # The output values should all be continuous except vuv (in acoustic model)
            print('computing norm stats for train_y...')
            if self.model_output_type == 'acoustic':
                vuv_index = self.out_streams.index('vuv')
                index = [sum([int(num) for num in self.outdims[0:vuv_index]])]
            else:
                index = []
            self.out_scaler = data_utils.compute_norm_stats(train_y,
                                                            self.out_stats_file,
                                                            method=self.out_norm,
                                                            no_scaling_ind=index)  # For vuv (the first column)

    def train_keras_model(self):

        #### define the model ####
        if not self.sequential_training:
            self.keras_models.define_feedforward_model()
        elif self.stateful:
            self.keras_models.define_stateful_model(batch_size=self.batch_size, seq_length=self.seq_length)
        else:
            self.keras_models.define_sequence_model()

        # TODO: for large datasets, I might have to batch load the data to memory... I will cross that bridge when it comes
        #### load the data ####
        print('preparing train_x, train_y from input and output feature files...')
        train_x, train_y, train_flen = data_utils.read_data_from_file_list(self.inp_train_file_list,
                                                                           self.out_train_file_list,
                                                                           self.inp_dim, self.out_dim,
                                                                           sequential_training=self.sequential_training)
        print('preparing valid_x, valid_y from input and output feature files...')
        valid_x, valid_y, valid_flen = data_utils.read_data_from_file_list(self.inp_valid_file_list,
                                                                           self.out_valid_file_list,
                                                                           self.inp_dim, self.out_dim,
                                                                           sequential_training=self.sequential_training)

        #### normalize the data (the input and output scalers need to be already created) ####
        train_x = data_utils.norm_data(train_x, self.inp_scaler, sequential_training=self.sequential_training)
        train_y = data_utils.norm_data(train_y, self.out_scaler, sequential_training=self.sequential_training)
        valid_x = data_utils.norm_data(valid_x, self.inp_scaler, sequential_training=self.sequential_training)
        valid_y = data_utils.norm_data(valid_y, self.out_scaler, sequential_training=self.sequential_training)

        #### train the model ####
        print('training...')
        if not self.sequential_training:
            ### Train feedforward model ###

            self.keras_models.train_feedforward_model(train_x, train_y,
                                                      valid_x, valid_y,
                                                      batch_size=self.batch_size,
                                                      num_of_epochs=self.num_of_epochs,
                                                      shuffle_data=self.shuffle_data,
                                                      tensorboard_dir=self.plot_dir)
        else:
            ### Train recurrent model ###
            print(('training algorithm: %d' % (self.training_algo)))
            self.keras_models.train_sequence_model(train_x, train_y, valid_x,
                                                   valid_y, train_flen,
                                                   batch_size=self.batch_size,
                                                   num_of_epochs=self.num_of_epochs,
                                                   shuffle_data=self.shuffle_data,
                                                   training_algo=self.training_algo)

        #### store the model ####
        self.keras_models.save_model(self.json_model_file, self.h5_model_file)

    def test_keras_model(self):
        #### load the model ####
        self.keras_models.load_model(self.json_model_file, self.h5_model_file)

        #### load the data ####
        print('preparing test_x from input feature files...')
        test_x, test_flen = data_utils.read_test_data_from_file_list(self.inp_test_file_list, self.inp_dim)

        #### normalize the data ####
        data_utils.norm_data(test_x, self.inp_scaler)

        #### compute predictions ####
        self.keras_models.predict(test_x, self.out_scaler, self.gen_test_file_list, self.sequential_training)

    def main_function(self):

        ### Implement each module ###
        if self.MAKELAB:
            self.make_labels()

        if self.MAKECMP:
            self.make_cmp()

        if self.NORMDATA:
            self.normalize_data()

        if self.TRAINDNN:
            self.train_keras_model()

        if self.TESTDNN:
            self.test_keras_model()

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print('usage: python run_keras_with_merlin_io.py [config file name]')
        sys.exit(1)

    # create a configuration instance
    # and get a short name for this instance
    cfg = configuration.configuration()

    config_file = sys.argv[1]

    config_file = os.path.abspath(config_file)
    cfg.configure(config_file)

    print("--- Job started ---")
    start_time = time.time()

    # main function
    keras_instance = KerasClass(cfg)
    keras_instance.main_function()

    (m, s) = divmod(int(time.time() - start_time), 60)
    print(("--- Job completion time: %d min. %d sec ---" % (m, s)))

    sys.exit(0)
