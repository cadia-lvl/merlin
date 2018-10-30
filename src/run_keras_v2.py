from builtins import map
from builtins import str
from builtins import range
import os
import sys
import random
import time
import matplotlib.pyplot as plt
import numpy as np

import configuration
from io_funcs.binary_io import BinaryIOCollection
from frontend.label_normalisation import HTSLabelNormalisation
from frontend.merge_features import MergeFeat
from frontend.label_normalisation import HTSLabelNormalisation, XMLLabelNormalisation
from frontend.silence_remover import SilenceRemover
from frontend.silence_remover import trim_silence
from frontend.min_max_norm import MinMaxNormalisation
from frontend.acoustic_composition import AcousticComposition
from frontend.parameter_generation import ParameterGeneration
from frontend.mean_variance_norm import MeanVarianceNorm

import logging.config
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from run_keras_with_merlin_io import \
    kerasModels,\
    read_file_list,\
    prepare_file_path_list,\
    read_data_from_file_list, \
    print_status


# class KerasBatchGenerator(object):
#
#     def __init__(self, data, batch_size):
#         self.data = data
#         # self.num_steps = num_steps
#         self.batch_size = batch_size
#         self.current_idx = 0
#
#     def generate(self):
#         x = np.zeros((self.batch_size, self.num_steps))
#         y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
#
#         for i in range(self.batch_size):
#             if self.current_idx + self.num_steps >= len(self.data):
#                 # reset the index back to the start of the data set
#                 self.current_idx = 0
#             x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
#             temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
#             # convert all of temp_y into a one hot representation
#             y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
#             self.current_idx += self.skip_step
#
#         yield x, y

def data_prep(cfg, label_normaliser):

    lab_dim = 0
    label_data_dir = 0
    in_label_align_file_list = 0
    binary_label_file_list = 0
    data_dir =0
    suffix =0
    nn_label_file_list=0
    nn_label_norm_file_list=0
    binary_label_dir = 0
    nn_label_dir = 0
    dur_file_list = 0


    min_max_normaliser = None
    label_norm_file = 'label_norm_%s_%d.dat' % (cfg.label_style, lab_dim)
    label_norm_file = os.path.join(label_data_dir, label_norm_file)

    # Collect file path lists for TEST set (label_align, binary_label, nn_label, nn_label_norm)
    if cfg.GenTestList:
        try:
            test_id_list = read_file_list(cfg.test_id_scp)
            logger.debug('Loaded file id list from %s' % cfg.test_id_scp)
        except IOError:
            # this means that open(...) threw an error
            logger.critical('Could not load file id list from %s' % cfg.test_id_scp)
            raise

        in_label_align_file_list = prepare_file_path_list(test_id_list, cfg.in_label_align_dir, cfg.lab_ext, False)
        binary_label_file_list = prepare_file_path_list(test_id_list, binary_label_dir, cfg.lab_ext)
        nn_label_file_list = prepare_file_path_list(test_id_list, nn_label_dir, cfg.lab_ext)
        nn_label_norm_file_list = prepare_file_path_list(test_id_list, nn_label_norm_dir, cfg.lab_ext)

    if cfg.NORMLAB and (cfg.label_style == 'HTS'):
        # simple HTS labels
        logger.info('preparing label data (input) using standard HTS style labels')
        label_normaliser.perform_normalisation(in_label_align_file_list, binary_label_file_list,
                                               label_type=cfg.label_type)

        if cfg.additional_features:
            out_feat_dir = os.path.join(data_dir, 'binary_label_' + suffix)
            out_feat_file_list = prepare_file_path_list(file_id_list, out_feat_dir, cfg.lab_ext)
            in_dim = label_normaliser.dimension
            for new_feature, new_feature_dim in cfg.additional_features.items():
                new_feat_dir = os.path.join(data_dir, new_feature)
                new_feat_file_list = prepare_file_path_list(file_id_list, new_feat_dir, '.' + new_feature)

                merger = MergeFeat(lab_dim=in_dim, feat_dim=new_feature_dim)
                merger.merge_data(binary_label_file_list, new_feat_file_list, out_feat_file_list)
                in_dim += new_feature_dim

                binary_label_file_list = out_feat_file_list

        remover = SilenceRemover(n_cmp=lab_dim, silence_pattern=cfg.silence_pattern, label_type=cfg.label_type,
                                 remove_frame_features=cfg.add_frame_features, subphone_feats=cfg.subphone_feats)
        remover.remove_silence(binary_label_file_list, in_label_align_file_list, nn_label_file_list)

        min_max_normaliser = MinMaxNormalisation(feature_dimension=lab_dim, min_value=0.01, max_value=0.99)
        # use only training data to find min-max information, then apply on the whole dataset
        if cfg.GenTestList:
            min_max_normaliser.load_min_max_values(label_norm_file)
        else:
            min_max_normaliser.find_min_max_values(nn_label_file_list[0:cfg.train_file_number])
        #  enforce silence such that the normalization runs without removing silence: only for final synthesis
        if cfg.GenTestList and cfg.enforce_silence:
            min_max_normaliser.normalise_data(binary_label_file_list, nn_label_norm_file_list)
        else:
            min_max_normaliser.normalise_data(nn_label_file_list, nn_label_norm_file_list)

    # save label normalisation information for unseen testing labels
    if min_max_normaliser is not None and not cfg.GenTestList:
        label_min_vector = min_max_normaliser.min_vector
        label_max_vector = min_max_normaliser.max_vector
        label_norm_info = np.concatenate((label_min_vector, label_max_vector), axis=0)

        label_norm_info = np.array(label_norm_info, 'float32')
        fid = open(label_norm_file, 'wb')
        label_norm_info.tofile(fid)
        fid.close()
        logger.info('saved %s vectors to %s' % (label_min_vector.size, label_norm_file))

    ### make output duration data
    if cfg.MAKEDUR:
        logger.info('creating duration (output) features')
        label_type = cfg.label_type
        feature_type = cfg.dur_feature_type
        label_normaliser.prepare_dur_data(in_label_align_file_list, dur_file_list, label_type, feature_type)

    ### make output acoustic data
    if cfg.MAKECMP:
        logger.info('creating acoustic (output) features')
        delta_win = cfg.delta_win  # [-0.5, 0.0, 0.5]
        acc_win = cfg.acc_win  # [1.0, -2.0, 1.0]

        acoustic_worker = AcousticComposition(delta_win=delta_win, acc_win=acc_win)
        if 'dur' in list(cfg.in_dir_dict.keys()) and cfg.AcousticModel:
            acoustic_worker.make_equal_frames(dur_file_list, lf0_file_list, cfg.in_dimension_dict)
        acoustic_worker.prepare_nn_data(in_file_list_dict, nn_cmp_file_list, cfg.in_dimension_dict,
                                        cfg.out_dimension_dict)

        if cfg.remove_silence_using_binary_labels:
            ## do this to get lab_dim:
            label_composer = LabelComposer()
            label_composer.load_label_configuration(cfg.label_config_file)
            lab_dim = label_composer.compute_label_dimension()

            silence_feature = 0  ## use first feature in label -- hardcoded for now
            logger.info('Silence removal from CMP using binary label file')

            ## overwrite the untrimmed audio with the trimmed version:
            trim_silence(nn_cmp_file_list, nn_cmp_file_list, cfg.cmp_dim,
                         binary_label_file_list, lab_dim, silence_feature)

        else:  ## back off to previous method using HTS labels:
            remover = SilenceRemover(n_cmp=cfg.cmp_dim, silence_pattern=cfg.silence_pattern, label_type=cfg.label_type,
                                     remove_frame_features=cfg.add_frame_features, subphone_feats=cfg.subphone_feats)
            remover.remove_silence(nn_cmp_file_list[0:cfg.train_file_number + cfg.valid_file_number],
                                   in_label_align_file_list[0:cfg.train_file_number + cfg.valid_file_number],
                                   nn_cmp_file_list[0:cfg.train_file_number + cfg.valid_file_number])  # save to itself


if __name__ == "__main__":

    start_time = time.time()

    # set up logging to use our custom class
    logger = logging.getLogger("main")

    # Configuration object cfg from config argument
    cfg = configuration.cfg
    cfg.configure(sys.argv[1])

    # Get training file id list
    file_id_list = read_file_list(cfg.file_id_scp)

    # Y data file lists
    # nn_cmp_dir = os.path.join(cfg.data_dir, 'nn' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim))
    nn_cmp_norm_dir = os.path.join(cfg.data_dir, 'nn_norm' + cfg.combined_feature_name + '_' + str(cfg.cmp_dim))
    # nn_cmp_file_list = prepare_file_path_list(file_id_list, nn_cmp_dir, cfg.cmp_ext)
    # nn_cmp_norm_file_list = prepare_file_path_list(file_id_list, nn_cmp_norm_dir, cfg.cmp_ext)

    # Get label dimensions
    label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name,
                                             add_frame_features=cfg.add_frame_features,
                                             subphone_feats=cfg.subphone_feats)
    add_feat_dim = sum(cfg.additional_features.values())
    inp_dim = label_normaliser.dimension + add_feat_dim + cfg.appended_input_dim
    out_dim = cfg.cmp_dim

    # X data file lists
    # binary_label_dir = os.path.join(cfg.data_dir, 'binary_label_'+str(label_normaliser.dimension))
    # nn_label_dir = os.path.join(cfg.data_dir, 'nn_no_silence_lab_'+str(lab_dim))
    nn_label_norm_dir = os.path.join(cfg.data_dir, 'nn_no_silence_lab_norm_'+str(inp_dim))
    # binary_label_file_list = prepare_file_path_list(file_id_list, binary_label_dir, cfg.lab_ext)
    # nn_label_file_list = prepare_file_path_list(file_id_list, nn_label_dir, cfg.lab_ext)
    # nn_label_norm_file_list = prepare_file_path_list(file_id_list, nn_label_norm_dir, cfg.lab_ext)

    # Split files into train and test sets
    train_file_number = cfg.train_file_number
    # valid_split = cfg.valid_split
    valid_file_number = cfg.valid_file_number
    test_file_number = cfg.test_file_number

    train_list = file_id_list[0:train_file_number]
    valid_list = file_id_list[train_file_number:train_file_number+valid_file_number]
    test_list = file_id_list[train_file_number+valid_file_number:train_file_number+valid_file_number+test_file_number]

    # Generate file path lists
    inp_train_file_list = prepare_file_path_list(train_list, nn_label_norm_dir, cfg.lab_ext)
    inp_valid_file_list = prepare_file_path_list(valid_list, nn_label_norm_dir, cfg.lab_ext)
    inp_test_file_list = prepare_file_path_list(test_list, nn_label_norm_dir, cfg.lab_ext)

    out_train_file_list = prepare_file_path_list(train_list, nn_cmp_norm_dir, cfg.dur_ext)
    out_valid_file_list = prepare_file_path_list(valid_list, nn_cmp_norm_dir, cfg.dur_ext)
    out_test_file_list = prepare_file_path_list(test_list, nn_cmp_norm_dir, cfg.dur_ext)

    # set to True if training recurrent models
    sequential_training = cfg.sequential_training

    # set to True if data to be shuffled
    shuffle_data = cfg.shuffle_data

    print 'preparing train_x, train_y from input and output feature files...'
    train_x, train_y, train_flen = read_data_from_file_list(inp_train_file_list, out_train_file_list, inp_dim, out_dim,
                                                            sequential_training=sequential_training)
    print 'preparing valid_x, valid_y from input and output feature files...'
    valid_x, valid_y, valid_flen = read_data_from_file_list(inp_valid_file_list, out_valid_file_list, inp_dim, out_dim,
                                                            sequential_training=sequential_training)  # We do simple
    print 'preparing test_x, test_y from input and output feature files...'
    test_x, test_y, test_flen = read_data_from_file_list(inp_test_file_list, out_test_file_list, inp_dim, out_dim,
                                                         sequential_training=sequential_training)

    # Set up DNN hyperparams
    optimizer = cfg.optimizer
    output_activation = cfg.output_activation
    loss_function = cfg.loss_function
    num_epochs = cfg.training_epochs
    batch_size = cfg.batch_size
    dropout_rate = cfg.dropout_rate
    l2_reg = cfg.l2_reg
    l1_reg = cfg.l1_reg
    hidden_layer_type = cfg.hidden_layer_type
    hidden_layer_size = cfg.hidden_layer_size

    # Train and or test
    train_model = cfg.train_model
    test_model = cfg.test_model

    # Build the model from the hyperparams
    dnn = kerasModels(n_in=inp_dim,
                      hidden_layer_size=hidden_layer_size,
                      n_out=out_dim,
                      hidden_layer_type=hidden_layer_type,
                      output_type=output_activation,
                      dropout_rate=dropout_rate,
                      loss_function=loss_function,
                      l1_reg=l1_reg,
                      l2_reg=l1_reg,
                      optimizer=optimizer)

    # Build the model name
    if sequential_training:
        combined_model_arch = 'RNN'
    else:
        combined_model_arch = 'DNN'
    combined_model_arch += '_' + '_'.join(map(str, hidden_layer_size))
    combined_model_arch += '_' + '_'.join(map(str, hidden_layer_type))
    nnets_file_name = '%s_%d_train_%d_%d_%d_%d_%d_model' \
                      % (combined_model_arch, int(shuffle_data),
                         inp_dim, out_dim, train_file_number, batch_size, num_epochs)

    # Extract model directory (for reading or writing to model files)
    model_dir = os.path.join(cfg.work_dir, 'nnets_model')
    gen_dir = os.path.join(cfg.work_dir, 'gen')
    json_model_file = os.path.join(model_dir, nnets_file_name+'.json')
    h5_model_file = os.path.join(model_dir, nnets_file_name+'.h5')

    print 'model file    : ' + nnets_file_name

    # Identify the prediction directory (for synthesis)
    pred_feat_dir = os.path.join(gen_dir, nnets_file_name)
    if not os.path.exists(pred_feat_dir):
        os.makedirs(pred_feat_dir)
    gen_test_file_list = prepare_file_path_list(test_list, pred_feat_dir, cfg.cmp_ext)
    gen_wav_file_list = prepare_file_path_list(test_list, pred_feat_dir, '.wav')

    # ------------------------ Model loading or training. --------------------------
    if not train_model:

        # load the model
        assert os.path.isfile(json_model_file), '.json model file noes not exist'
        assert os.path.isfile(h5_model_file), '.h5 weight file noes not exist'

        json_file = open(json_model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(h5_model_file)
        print("Loaded model from disk")

        # compile the model
        dnn.model = loaded_model
        dnn.compile_model()
        model = dnn.model

    else:  # Instantiate and train

        early_stopping = EarlyStopping(monitor='val_loss', patience=cfg.stopping_patience)

        if sequential_training:  # Recurrent model

            dnn.define_sequence_model()
            if batch_size == 1:

                # train each sentence as a batch (I think this is the better way: we do not want a stateful model)
                train_index_list = list(range(train_file_number))
                valid_index_list = list(range(valid_file_number))
                training_loss = np.zeros((num_epochs,))
                validation_loss = np.zeros((num_epochs,))

                if shuffle_data:
                    random.seed(271638)
                    random.shuffle(train_index_list)

                for epoch_num in range(num_epochs):

                    print 'Epoch: %d/%d ' % (epoch_num + 1, num_epochs)
                    utt_count = -1

                    # Train and collect the loss data
                    batch_losses = np.zeros((len(train_index_list),))
                    for utt_index in train_index_list:
                        temp_train_x = train_x[train_list[utt_index]]
                        temp_train_y = train_y[train_list[utt_index]]
                        temp_train_x = np.reshape(temp_train_x, (1, temp_train_x.shape[0], inp_dim))
                        temp_train_y = np.reshape(temp_train_y, (1, temp_train_y.shape[0], out_dim))
                        batch_losses[utt_index] = dnn.model.train_on_batch(temp_train_x, temp_train_y)[0]
                        utt_count += 1
                        print_status(utt_count, train_file_number)
                    training_loss[epoch_num] = np.mean(batch_losses)

                    # Evaluate on the validation dataset and collect loss
                    batch_losses = np.zeros((len(valid_index_list),))
                    count = 0
                    for utt_index in valid_index_list:
                        temp_valid_x = valid_x[valid_list[utt_index]]
                        temp_valid_y = valid_y[valid_list[utt_index]]
                        temp_valid_x = np.reshape(temp_valid_x, (1, temp_valid_x.shape[0], inp_dim))
                        temp_valid_y = np.reshape(temp_valid_y, (1, temp_valid_y.shape[0], out_dim))
                        batch_losses[count] = dnn.model.test_on_batch(temp_valid_x, temp_valid_y)[0]
                        count += 1
                    validation_loss[epoch_num] = np.mean(batch_losses)

                    sys.stdout.write("\n")

            else:
                # if batch size more than 1
                train_count_list = list(train_flen.keys())
                if shuffle_data:
                    random.seed(271638)
                    random.shuffle(train_count_list)
                for epoch_num in range(num_epochs):
                    print 'Epoch: %d/%d ' % (epoch_num + 1, num_epochs)
                    utt_count = -1
                    for frame_number in train_count_list:
                        batch_file_list = train_flen[frame_number]
                        num_of_files = len(batch_file_list)
                        temp_train_x = np.zeros((num_of_files, frame_number, inp_dim))
                        temp_train_y = np.zeros((num_of_files, frame_number, out_dim))
                        for file_index in range(num_of_files):
                            temp_train_x[file_index,] = train_x[batch_file_list[file_index]]
                            temp_train_y[file_index,] = train_y[batch_file_list[file_index]]
                            history = dnn.model.fit(temp_train_x, temp_train_y,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    epochs=1,
                                                    verbose=1,
                                                    validation_split=valid_split,
                                                    callbacks=[early_stopping])
                        utt_count += num_of_files
                        print_status(utt_count, train_file_number)

                    sys.stdout.write("\n")

        else:
            # Train baseline (FF) model
            dnn.define_baseline_model()
            history = dnn.model.fit(train_x, train_y,
                                    batch_size=batch_size,
                                    epochs=num_epochs,
                                    shuffle=shuffle_data,
                                    validation_data=(valid_x, valid_y),
                                    callbacks=[early_stopping])


    # store the model

    # serialize model to JSON
    model_json = dnn.model.to_json()
    with open(json_model_file, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    dnn.model.save_weights(h5_model_file)
    print('\n')
    print('saved json model to: %s' % json_file)
    print('saved weights model to: %s' % h5_model_file)

    (m, s) = divmod(int(time.time() - start_time), 60)
    print("--- Job completion time: %d min. %d sec ---" % (m, s))

    # Plot training and validation errors
    plt.figure(1)
    if sequential_training:
        plt.plot(training_loss)
        plt.plot(validation_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    else:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    print " "