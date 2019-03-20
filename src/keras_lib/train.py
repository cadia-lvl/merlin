from __future__ import division
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

from builtins import str
from builtins import range
from past.utils import old_div
import os, sys
import random
import numpy as np
from keras import callbacks

from keras_lib.model import kerasModels
from keras_lib import data_utils
from keras_lib.data_sequence import UttBatchSequence

from io_funcs.binary_io import BinaryIOCollection


def named_logs(logs):
  result = {}
  for l in logs:
    result['loss'] = l
  return result


class TrainKerasModels(kerasModels):

    def __init__(self, model_params, rnn_params, training_params):

        # Subclass Keras model, feed in model parameters
        kerasModels.__init__(self, model_params)

        # Rnn parameters
        self.merge_size = rnn_params['merge_size']
        self.seq_length = rnn_params['seq_length']
        self.bucket_range = rnn_params['bucket_range']
        self.stateful = rnn_params['stateful']

        # Training parameters
        self.batch_size = training_params['batch_size']
        self.num_of_epochs = training_params['num_of_epochs']
        self.shuffle_data = training_params['shuffle_data']
        self.tensorboard_dir = training_params['tensorboard_dir']
        self.stopping_patience = training_params['stopping_patience']
        self.restore_best_weights = training_params['restore_best_weights']

    def train_shared_model(self, train_x, train_y, valid_x, valid_y):

        tb_callback_dict = {speaker: None for speaker in self.speaker_id}
        for model in self.models:
            # Set up tensorboard
            tb_callback_dict[model.name] = callbacks.TensorBoard(log_dir=self.tensorboard_dir,
                                                                 histogram_freq=0,
                                                                 write_graph=True,
                                                                 write_grads=True,
                                                                 write_images=False,
                                                                 batch_size=self.batch_size)
            tb_callback_dict[model.name].set_model(model)

        # Need to randomize batches
        train_id_list = list(train_x.keys())
        valid_id_list = list(valid_x.keys())
        if self.shuffle_data:
            random.seed(271638)
            random.shuffle(train_id_list)

        train_file_number = len(train_x)
        valid_file_number = len(valid_x)
        training_loss = {speaker: [] for speaker in self.speaker_id}
        validation_loss = {speaker: [] for speaker in self.speaker_id}
        for epoch_num in range(self.num_of_epochs):

            print(('\nEpoch: %d/%d ' %(epoch_num+1, self.num_of_epochs)))
            batch_training_loss = {speaker: [] for speaker in self.speaker_id}
            batch_validation_loss = {speaker: [] for speaker in self.speaker_id}
            batch_count = {speaker: 0 for speaker in self.speaker_id}
            for i in range(train_file_number):

                key = train_id_list[i]
                x = train_x[key].reshape(1, train_x[key].shape[0], self.inp_dim)
                y = train_y[key].reshape(1, train_y[key].shape[0], self.out_dim)

                # Identify which output to use
                ind = np.where([spk in key for spk in self.speaker_id])[0][0]
                batch_speaker = self.speaker_id[ind]

                # Run train on batch
                for model in self.models:
                    if model.name == batch_speaker:
                        batch_training_loss[batch_speaker].append(model.train_on_batch(x, y))
                        tb_callback_dict[batch_speaker].on_batch_end(batch_count[batch_speaker],
                                                                     {'loss': batch_training_loss[batch_speaker][-1]})
                        batch_count[batch_speaker] += 1

                data_utils.drawProgressBar(i, train_file_number-1)

            # Average training loss per epoch
            for speaker in batch_training_loss.keys():
                training_loss[speaker].append(np.mean(batch_training_loss[speaker]))
                print('\nTraining loss %s: %.3f' % (speaker, training_loss[speaker][-1]))
                tb_callback_dict[speaker].on_epoch_end(epoch_num, {'loss': training_loss[speaker][-1]})

            # for each epoch, run validation set
            for i in range(valid_file_number):
                key = valid_id_list[i]
                x = valid_x[key].reshape(1, valid_x[key].shape[0], self.inp_dim)
                y = valid_y[key].reshape(1, valid_y[key].shape[0], self.out_dim)

                # Identify which output to use
                ind = np.where([spk in key for spk in self.speaker_id])[0][0]
                batch_speaker = self.speaker_id[ind]

                # Run test on batch
                for model in self.models:
                    if model.name == batch_speaker:
                       batch_validation_loss[batch_speaker].append(model.test_on_batch(x, y))

                data_utils.drawProgressBar(i, valid_file_number-1)

            # Average validation loss per epoch
            for speaker in batch_training_loss.keys():
                validation_loss[speaker].append(np.mean(batch_validation_loss[speaker]))
                print('\nValidation loss %s: %.3f' % (speaker, validation_loss[speaker][-1]))

        # Signal end of training to tensorboard
        for speaker in tb_callback_dict.keys():
            tb_callback_dict[speaker].on_train_end(None)

    def train_feedforward_model(self, train_x, train_y, valid_x, valid_y):

        # Set up callbacks
        tb_callback = callbacks.TensorBoard(log_dir=self.tensorboard_dir, histogram_freq=1, write_graph=True,
                                            write_images=False, batch_size=self.batch_size)
        es_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.stopping_patience,
                                              verbose=0, mode='auto', baseline=None,
                                              restore_best_weights=self.restore_best_weights)

        # Train the model
        self.model.fit(x=train_x, y=train_y, validation_data=(valid_x, valid_y),
                       batch_size=self.batch_size, epochs=self.num_of_epochs, shuffle=self.shuffle_data,
                       callbacks=[tb_callback, es_callback])

    # def train_sequence_model(self, train_x, train_y, valid_x, valid_y, train_flen, batch_size=1, num_of_epochs=10, shuffle_data=True, training_algo=1):
    #     # TODO: use packaged params
    #
    #     if batch_size == 1:
    #         self.train_recurrent_model_batchsize_one(train_x, train_y, valid_x, valid_y)
    #     else:
    #         self.train_recurrent_model(train_x, train_y, valid_x, valid_y, train_flen, batch_size, num_of_epochs, shuffle_data, training_algo)

    def train_recurrent_model_batchsize_one(self, train_x, train_y, valid_x, valid_y):

        # Set up callbacks
        tb_callback = callbacks.TensorBoard(log_dir=self.tensorboard_dir, write_graph=True)
        es_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.stopping_patience,
                                              verbose=0, mode='auto', baseline=None,
                                              restore_best_weights=self.restore_best_weights)

        train_sequence = UttBatchSequence(list(train_x.values()), list(train_y.values()))
        valid_sequence = UttBatchSequence(list(valid_x.values()), list(valid_y.values()))

        self.model.fit_generator(train_sequence, epochs=self.num_of_epochs, verbose=1,
                                 callbacks=[tb_callback, es_callback], validation_data=valid_sequence,
                                 workers=1, use_multiprocessing=False, shuffle=self.shuffle_data)

        # # if batch size is equal to 1
        # train_idx_list = list(train_x.keys())
        # valid_idx_list = list(valid_x.keys())
        # if self.shuffle_data:
        #     random.seed(271638)
        #     random.shuffle(train_idx_list)
        #
        # train_file_number = len(train_idx_list)
        # for epoch_num in range(self.num_of_epochs):
        #     print(('Epoch: %d/%d ' % (epoch_num+1, self.num_of_epochs)))
        #     file_num = 0
        #
        #     # Train
        #     for file_name in train_idx_list:
        #         temp_train_x = train_x[file_name]
        #         temp_train_y = train_y[file_name]
        #         temp_train_x = np.reshape(temp_train_x, (1, temp_train_x.shape[0], self.inp_dim))
        #         temp_train_y = np.reshape(temp_train_y, (1, temp_train_y.shape[0], self.out_dim))
        #         self.model.train_on_batch(temp_train_x, temp_train_y)
        #         file_num += 1
        #         data_utils.drawProgressBar(file_num, train_file_number)
        #
        #     # Validate
        #     error = np.empty(shape=(len(valid_idx_list), 1))
        #     accuracy = np.empty(shape=(len(valid_idx_list), 1))
        #     for i, file_name in enumerate(valid_idx_list):
        #         temp_valid_x = valid_x[file_name]
        #         temp_valid_y = valid_y[file_name]
        #         temp_valid_x = np.reshape(temp_valid_x, (1, temp_valid_x.shape[0], self.inp_dim))
        #         temp_valid_y = np.reshape(temp_valid_y, (1, temp_valid_y.shape[0], self.out_dim))
        #         error[i], accuracy[i] = self.model.test_on_batch(temp_valid_x, temp_valid_y)
        #
        #     print('validation error: %.3f \nvalidation accuracy: %.3f' % (np.mean(error), np.mean(accuracy)))
        #     sys.stdout.write("\n")

    def train_recurrent_model(self, train_x, train_y, valid_x, valid_y, train_flen, training_algo):
        # TODO: use packaged params
        ### if batch size more than 1 ###
        if training_algo == 1:
            self.train_padding_model(train_x, train_y, valid_x, valid_y, train_flen)
        elif training_algo == 2:
            self.train_bucket_model(train_x, train_y, valid_x, valid_y, train_flen)
        elif training_algo == 3:
            self.train_split_model(train_x, train_y, valid_x, valid_y, train_flen)
        else:
            print("Choose training algorithm for batch training with RNNs:")
            print("1. Padding model -- pad utterances with zeros to maximum sequence length")
            print("2. Bucket model  -- form buckets with minimum and maximum sequence length")
            print("3. Split model   -- split utterances to a fixed sequence length")
            sys.exit(1)

    def train_padding_model(self, train_x, train_y, valid_x, valid_y, train_flen):
        # TODO: use packaged params
        ### Method 1 ###
        train_id_list = list(train_flen['utt2framenum'].keys())
        if self.shuffle_data:
            random.seed(271638)
            random.shuffle(train_id_list)

        train_file_number = len(train_id_list)
        for epoch_num in range(self.num_of_epochs):
            print(('Epoch: %d/%d ' %(epoch_num+1, self.num_of_epochs)))
            file_num = 0
            while file_num < train_file_number:
                train_idx_list = train_id_list[file_num: file_num + self.batch_size]
                seq_len_arr    = [train_flen['utt2framenum'][filename] for filename in train_idx_list]
                max_seq_length = max(seq_len_arr)
                sub_train_x    = dict((filename, train_x[filename]) for filename in train_idx_list)
                sub_train_y    = dict((filename, train_y[filename]) for filename in train_idx_list)
                temp_train_x   = data_utils.transform_data_to_3d_matrix(sub_train_x, max_length=max_seq_length)
                temp_train_y   = data_utils.transform_data_to_3d_matrix(sub_train_y, max_length=max_seq_length)
                self.model.train_on_batch(temp_train_x, temp_train_y)
                file_num += len(train_idx_list)
                data_utils.drawProgressBar(file_num, train_file_number)

            print(" Validation error: %.3f" % (self.get_validation_error(valid_x, valid_y)))
    
    def train_bucket_model(self, train_x, train_y, valid_x, valid_y, train_flen):
        # TODO: use packaged params
        ### Method 2 ###
        train_fnum_list  = np.array(list(train_flen['framenum2utt'].keys()))
        train_range_list = list(range(min(train_fnum_list), max(train_fnum_list)+1, self.bucket_range))
        if self.shuffle_data:
            random.seed(271638)
            random.shuffle(train_range_list)

        train_file_number = len(train_x)
        for epoch_num in range(self.num_of_epochs):
            print(('Epoch: %d/%d ' %(epoch_num+1, self.num_of_epochs)))
            file_num = 0
            for frame_num in train_range_list:
                min_seq_length = frame_num
                max_seq_length = frame_num+self.bucket_range
                sub_train_list = train_fnum_list[(train_fnum_list>=min_seq_length) & (train_fnum_list<max_seq_length)]
                if len(sub_train_list)==0:
                    continue;
                train_idx_list = sum([train_flen['framenum2utt'][framenum] for framenum in sub_train_list], [])
                sub_train_x    = dict((filename, train_x[filename]) for filename in train_idx_list)
                sub_train_y    = dict((filename, train_y[filename]) for filename in train_idx_list)
                temp_train_x   = data_utils.transform_data_to_3d_matrix(sub_train_x, max_length=max_seq_length)
                temp_train_y   = data_utils.transform_data_to_3d_matrix(sub_train_y, max_length=max_seq_length)
                self.model.fit(temp_train_x, temp_train_y, batch_size=self.batch_size, shuffle=False, epochs=1, verbose=0)

                file_num += len(train_idx_list)
                data_utils.drawProgressBar(file_num, train_file_number)

            print(" Validation error: %.3f" % (self.get_validation_error(valid_x, valid_y)))

    def train_split_model(self, train_x, train_y, valid_x, valid_y, train_flen):
        # TODO: use packaged params
        ### Method 3 ###
        train_id_list = list(train_flen['utt2framenum'].keys())
        if self.shuffle_data:
            random.seed(271638)
            random.shuffle(train_id_list)

        train_file_number = len(train_id_list)
        for epoch_num in range(self.num_of_epochs):
            print(('Epoch: %d/%d ' %(epoch_num+1, self.num_of_epochs)))
            file_num = 0
            while file_num < train_file_number:
                train_idx_list = train_id_list[file_num: file_num + self.batch_size]
                sub_train_x    = dict((filename, train_x[filename]) for filename in train_idx_list)
                sub_train_y    = dict((filename, train_y[filename]) for filename in train_idx_list)
                temp_train_x   = data_utils.transform_data_to_3d_matrix(sub_train_x, seq_length=self.seq_length, merge_size=self.merge_size)
                temp_train_y   = data_utils.transform_data_to_3d_matrix(sub_train_y, seq_length=self.seq_length, merge_size=self.merge_size)
    
                self.model.train_on_batch(temp_train_x, temp_train_y)

                file_num += len(train_idx_list)
                data_utils.drawProgressBar(file_num, train_file_number)

            print(" Validation error: %.3f" % (self.get_validation_error(valid_x, valid_y)))

    def train_split_model_keras_version(self, train_x, train_y, valid_x, valid_y, train_flen, batch_size, num_of_epochs, shuffle_data):
        """This function is not used as of now 
        """
        ### Method 3 ###
        temp_train_x = data_utils.transform_data_to_3d_matrix(train_x, seq_length=self.seq_length, merge_size=self.merge_size, shuffle_data=shuffle_data)
        print(("Input shape: "+str(temp_train_x.shape)))
        
        temp_train_y = data_utils.transform_data_to_3d_matrix(train_y, seq_length=self.seq_length, merge_size=self.merge_size, shuffle_data=shuffle_data)
        print(("Output shape: "+str(temp_train_y.shape)))
        
        if self.stateful:
            temp_train_x, temp_train_y = data_utils.get_stateful_data(temp_train_x, temp_train_y, batch_size)
    
        self.model.fit(temp_train_x, temp_train_y, batch_size=batch_size, epochs=num_of_epochs)
    
    def train_bucket_model_without_padding(self, train_x, train_y, valid_x, valid_y, train_flen, batch_size, num_of_epochs, shuffle_data):
        """This function is not used as of now
        """
        ### Method 4 ###
        train_count_list = list(train_flen['framenum2utt'].keys())
        if shuffle_data:
            random.seed(271638)
            random.shuffle(train_count_list)

        train_file_number = len(train_x)
        for epoch_num in range(num_of_epochs):
            print(('Epoch: %d/%d ' %(epoch_num+1, num_of_epochs)))
            file_num = 0
            for sequence_length in train_count_list:
                train_idx_list = train_flen['framenum2utt'][sequence_length]
                sub_train_x    = dict((filename, train_x[filename]) for filename in train_idx_list)
                sub_train_y    = dict((filename, train_y[filename]) for filename in train_idx_list)
                temp_train_x   = data_utils.transform_data_to_3d_matrix(sub_train_x, max_length=sequence_length)
                temp_train_y   = data_utils.transform_data_to_3d_matrix(sub_train_y, max_length=sequence_length)
                self.model.fit(temp_train_x, temp_train_y, batch_size=batch_size, epochs=1, verbose=0)

                file_num += len(train_idx_list)
                data_utils.drawProgressBar(file_num, train_file_number)

            sys.stdout.write("\n")

    def get_validation_error(self, valid_x, valid_y, sequential_training=True, stateful=False):
        valid_id_list = list(valid_x.keys())
        valid_id_list.sort()

        valid_error = 0.0
        valid_file_number = len(valid_id_list)
        for utt_index in range(valid_file_number):
            temp_valid_x = valid_x[valid_id_list[utt_index]]
            temp_valid_y = valid_y[valid_id_list[utt_index]]
            num_of_rows = temp_valid_x.shape[0]

            if stateful:
                temp_valid_x = data_utils.get_stateful_input(temp_valid_x, self.seq_length, self.batch_size)
            elif sequential_training:
                temp_valid_x = np.reshape(temp_valid_x, (1, num_of_rows, self.inp_dim))

            predictions = self.model.predict(temp_valid_x)
            if sequential_training:
                predictions = np.reshape(predictions, (num_of_rows, self.out_dim))

            valid_error += np.mean(np.sum((predictions - temp_valid_y) ** 2, axis=1))

        valid_error = old_div(valid_error,valid_file_number)

        return valid_error

    def predict(self, test_x, out_scaler, gen_test_file_list, sequential_training=False, stateful=False):
        #### compute predictions ####
        io_funcs = BinaryIOCollection()

        test_file_number = len(gen_test_file_list)
        print("generating features on held-out test data...")
        for utt_index in range(test_file_number):
            gen_test_file_name = gen_test_file_list[utt_index]
            test_id = os.path.splitext(os.path.basename(gen_test_file_name))[0]
            temp_test_x        = test_x[test_id]
            num_of_rows        = temp_test_x.shape[0]

            if stateful:
                temp_test_x = data_utils.get_stateful_input(temp_test_x, self.seq_length, self.batch_size)
            elif sequential_training:
                temp_test_x = np.reshape(temp_test_x, (1, num_of_rows, self.inp_dim))

            predictions = self.model.predict(temp_test_x)
            if sequential_training:
                predictions = np.reshape(predictions, (num_of_rows, self.out_dim))

            data_utils.denorm_data(predictions, out_scaler)

            io_funcs.array_to_binary_file(predictions, gen_test_file_name)
            data_utils.drawProgressBar(utt_index+1, test_file_number)

        sys.stdout.write("\n")
