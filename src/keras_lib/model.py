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

from builtins import range
from builtins import object
import random
import numpy as np

import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, SimpleRNN, GRU, LSTM
from keras.layers import Dropout
from keras.regularizers import l1_l2
from keras.utils import multi_gpu_model

class kerasModels(object):

    def __init__(self, model_params):

        """ This function initialises a neural network

        :param n_in: Dimensionality of input features
        :param hidden_layer_size: The layer size for each hidden layer
        :param n_out: Dimensionality of output features
        :param hidden_layer_type: the activation types of each hidden layers, e.g., TANH, LSTM, GRU, BLSTM
        :param output_type: the activation type of the output layer, by default is 'LINEAR', linear regression.
        :param dropout_rate: probability of dropout, a float number between 0 and 1.
        :type n_in: Integer
        :type hidden_layer_size: A list of integers
        :type n_out: Integrer
        """

        self.inp_dim = int(model_params['inp_dim'])
        self.out_dim = int(model_params['out_dim'])
        self.n_layers = len(model_params['hidden_layer_size'])
        self.hidden_layer_size = model_params['hidden_layer_size']
        self.hidden_layer_type = model_params['hidden_layer_type']
        self.output_type = model_params['output_layer_type']
        self.dropout_rate = model_params['dropout_rate']
        self.loss_function = model_params['loss_function']
        self.l1 = model_params['l1']
        self.l2 = model_params['l2']
        self.optimizer = model_params['optimizer']
        self.gpu_num = model_params['gpu_num']

        assert len(self.hidden_layer_size) == len(self.hidden_layer_type)

        # create model
        self.model = Sequential()

    def define_feedforward_model(self):
        seed = 12345
        np.random.seed(seed)

        # add hidden layers
        for i in range(self.n_layers):
            if i == 0:
                input_size = self.inp_dim
            else:
                input_size = self.hidden_layer_size[i - 1]

            self.model.add(Dense(
                    units=self.hidden_layer_size[i],
                    activation=self.hidden_layer_type[i],
                    kernel_initializer="normal",
                    input_dim=input_size,
                    kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2)))

            self.model.add(Dropout(self.dropout_rate))

        # add output layer
        self.model.add(Dense(
            units=self.out_dim,
            activation=self.output_type.lower(),
            kernel_initializer="normal",
            input_dim=self.hidden_layer_size[-1]))

        # Compile the model
        self.compile_model()

    def define_sequence_model(self):
        seed = 12345
        np.random.seed(seed)

        # add hidden layers
        for i in range(self.n_layers):
            if i == 0:
                input_size = self.inp_dim
            else:
                input_size = self.hidden_layer_size[i - 1]

            if self.hidden_layer_type[i]=='rnn':
                self.model.add(SimpleRNN(
                        units=self.hidden_layer_size[i],
                        input_shape=(None, input_size),
                        return_sequences=True))
            elif self.hidden_layer_type[i]=='gru':
                self.model.add(GRU(
                        units=self.hidden_layer_size[i],
                        input_shape=(None, input_size),
                        return_sequences=True))
            elif self.hidden_layer_type[i]=='lstm':
                self.model.add(LSTM(
                        units=self.hidden_layer_size[i],
                        input_shape=(None, input_size),
                        return_sequences=True))
            elif self.hidden_layer_type[i]=='blstm':
                self.model.add(LSTM(
                        units=self.hidden_layer_size[i],
                        input_shape=(None, input_size),
                        return_sequences=True,
                        go_backwards=True))
            else:
                self.model.add(Dense(
                        units=self.hidden_layer_size[i],
                        activation=self.hidden_layer_type[i],
                        kernel_initializer="normal",
                        input_shape=(None, input_size)))

        # add output layer
        self.model.add(Dense(
            units=self.out_dim,
            input_dim=self.hidden_layer_size[-1],
            kernel_initializer='normal',
            activation=self.output_type.lower()))

        # Compile the model
        self.compile_model()

    def define_stateful_model(self, batch_size=25, seq_length=200):
        seed = 12345
        np.random.seed(seed)

        # params
        batch_size = batch_size
        timesteps  = seq_length

        # add hidden layers
        for i in range(self.n_layers):
            if i == 0:
                input_size = self.inp_dim
            else:
                input_size = self.hidden_layer_size[i - 1]

            if self.hidden_layer_type[i]=='lstm':
                self.model.add(LSTM(
                        units=self.hidden_layer_size[i],
                        batch_input_shape=(batch_size, timesteps, input_size),
                        return_sequences=True,
                        stateful=True))   #go_backwards=True))
            elif self.hidden_layer_type[i]=='blstm':
                self.model.add(LSTM(
                        units=self.hidden_layer_size[i],
                        batch_input_shape=(batch_size, timesteps, input_size),
                        return_sequences=True,
                        stateful=True,
                        go_backwards=True))
            else:
                self.model.add(Dense(
                        units=self.hidden_layer_size[i],
                        activation=self.hidden_layer_type[i],
                        kernel_initializer="normal",
                        batch_input_shape=(batch_size, timesteps, input_size)))

        # add output layer
        self.model.add(Dense(
            units=self.out_dim,
            input_dim=self.hidden_layer_size[-1],
            kernel_initializer='normal',
            activation=self.output_type.lower()))

        # Compile the model
        self.compile_model()

    def compile_model(self):

        # Parallelize gpus
        if self.gpu_num > 1:
            self.model = multi_gpu_model(self.model, gpus=self.gpu_num)

        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

    def save_model(self, json_model_file, h5_model_file):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(json_model_file, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(h5_model_file)
        print("Saved model to disk")

    def load_model(self, json_model_file, h5_model_file):
        #### load the model ####
        json_file = open(json_model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(h5_model_file)
        print("Loaded model from disk")

        #### compile the model ####
        self.model = loaded_model
        self.compile_model()
