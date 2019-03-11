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

from future import standard_library
standard_library.install_aliases()
from builtins import map
from builtins import str
from builtins import range
from builtins import object
import sys
if sys.version_info.major >= 3:
    import configparser
else:
    import configparser as configparser
import logging
import os

from frontend.label_normalisation import HTSLabelNormalisation

class configuration(object):

    def __init__(self):
        pass;

    def configure(self, configFile=None):

        # get a logger
        logger = logging.getLogger("configuration")
        # this (and only this) logger needs to be configured immediately, otherwise it won't work
        # we can't use the full user-supplied configuration mechanism in this particular case,
        # because we haven't loaded it yet!
        #
        # so, just use simple console-only logging
        logger.setLevel(logging.DEBUG) # this level is hardwired here - should change it to INFO
        # add a handler & its formatter - will write only to console
        ch = logging.StreamHandler()
        logger.addHandler(ch)
        formatter = logging.Formatter('%(asctime)s %(levelname)8s%(name)15s: %(message)s')
        ch.setFormatter(formatter)

        # first, set up some default configuration values
        self.initial_configuration()

        # next, load in any user-supplied configuration values
        # that might over-ride the default values
        self.user_configuration(configFile)

        # finally, set up all remaining configuration values
        # that depend upon either default or user-supplied values
        self.complete_configuration()

        logger.debug('configuration completed')

    def initial_configuration(self):

        # to be called before loading any user specific values

        # things to put here are
        # 1. variables that the user cannot change
        # 2. variables that need to be set before loading the user's config file

        UTTID_REGEX = '(.*)\..*'

    def user_configuration(self,configFile=None):

        # get a logger
        logger = logging.getLogger("configuration")

        # load and parse the provided configFile, if provided
        if not configFile:
            logger.warn('no user configuration file provided; using only built-in default settings')
            return

        # load the config file
        try:
            cfgparser = configparser.ConfigParser()
            cfgparser.readfp(open(configFile))
            logger.debug('successfully read and parsed user configuration file %s' % configFile)
        except:
            logger.fatal('error reading user configuration file %s' % configFile)
            raise

        #work_dir must be provided before initialising other directories
        try:
            self.work_dir = cfgparser.get('Paths', 'work')
            self.data_dir = cfgparser.get('Paths', 'data')
            self.plot_dir = cfgparser.get('Paths', 'plot')
            self.model_output_type = cfgparser.get('Input-Output', 'model_output_type')

        except (configparser.NoSectionError, configparser.NoOptionError):
            self.work_dir = None
            self.data_dir = None
            self.plot_dir = None
            logger.critical('Paths:work has no value!')
            raise Exception

        # The model must be placed in the processors folder which is copied to the voice folder for synthesis
        if self.model_output_type == 'duration':
            self.model_dir = os.path.join(self.data_dir, 'processors', 'duration_predictor')
        elif self.model_output_type == 'acoustic':
            self.model_dir = os.path.join(self.data_dir, 'processors', 'acoustic_predictor')

        # default place for some data
        self.keras_dir = os.path.join(self.work_dir, 'keras')
        self.gen_dir = os.path.join(self.keras_dir, 'gen')
        self.stats_dir = os.path.join(self.keras_dir, 'stats')

        self.question_file_name = cfgparser.get('Labels', 'question_file_name')
        self.add_frame_features = cfgparser.get('Labels', 'add_frame_features')
        self.subphone_feats = cfgparser.get('Labels', 'subphone_feats')

        self.model_type = cfgparser.get('Labels', 'subphone_feats')

        # TODO: the configuration is inflexible, it has hard coded elements and is designed only for the acoustic model
        # TODO: improve flexibility and incorporate duration model elements
        # TODO: I am going to perform all data normalization tasks in KerasClass

        # Set up file paths to ossian defaults
        label_normaliser = HTSLabelNormalisation(question_file_name=self.question_file_name,
                                                 add_frame_features=self.add_frame_features,
                                                 subphone_feats=self.subphone_feats)
        self.inp_dim = label_normaliser.dimension
        # lab_dim = label_normaliser.dimension
        # logger.info('Input label dimension is %d' % lab_dim)
        # suffix = str(lab_dim)

        # the number can be removed
        # binary_label_dir = os.path.join(self.work_dir, 'binary_label_' + str(label_normaliser.dimension))
        # nn_label_dir = os.path.join(self.work_dir, 'nn_no_silence_lab_' + suffix)
        # self.def_inp_dir = os.path.join(self.work_dir, 'nn_no_silence_lab_norm_' + suffix)

        # self.def_inp_dir = os.path.join(self.work_dir, 'nn_no_silence_lab_norm_%s' % 1)
        # self.def_out_dir = os.path.join(self.work_dir, 'nn_norm_mgc_lf0_vuv_bap_%s' % 1)

        # ---------------------------------------------------
        # ------------------- Output data -------------------
        # ---------------------------------------------------
        # Binary data (already generated by ossian)
        self.out_feat_dir_dur = os.path.join(self.data_dir, 'dur')
        self.out_feat_dir_cmp = os.path.join(self.data_dir, 'cmp')

        self.out_feat_dir_dur_norm = os.path.join(self.data_dir, 'dur_norm')
        self.out_feat_dir_cmp_norm = os.path.join(self.data_dir, 'cmp_norm')

        self.nn_cmp_dir = os.path.join(self.data_dir, 'nn_cmp')
        self.nn_cmp_norm_dir = os.path.join(self.data_dir, 'nn_norm_cmp')

        # ---------------------------------------------------
        # ------------------- Input data -------------------
        # ---------------------------------------------------
        # Raw text data
        self.inp_feat_dir_dur = os.path.join(self.data_dir, 'lab_dur')
        self.inp_feat_dir_cmp = os.path.join(self.data_dir, 'lab_dnn')

        # Binary data
        self.bin_lab_dir_dur = os.path.join(self.data_dir, 'bin_lab_phone_%s' % str(self.inp_dim))
        self.bin_lab_dir_cmp = os.path.join(self.data_dir, 'bin_lab_state_%s' % str(self.inp_dim))

        # Binary data silence removed
        self.bin_lab_dir_dur_nosilence = os.path.join(self.data_dir, 'bin_lab_phone_no_sil_%s' % str(self.inp_dim))
        self.bin_lab_dir_cmp_nosilence = os.path.join(self.data_dir, 'bin_lab_state_no_sil_%s' % str(self.inp_dim))

        # Binary data silence removed and normalized
        self.bin_lab_dir_dur_nosilence_norm = os.path.join(self.data_dir, 'bin_lab_phone_no_sil_norm_%s' % str(self.inp_dim))
        self.bin_lab_dir_cmp_nosilence_norm = os.path.join(self.data_dir, 'bin_lab_state_no_sil_norm_%s' % str(self.inp_dim))

        # self.inter_data_dir = os.path.join(self.work_dir, 'inter_module')
        # self.def_inp_dir    = os.path.join(self.inter_data_dir, 'nn_no_silence_lab_norm_425')
        # self.def_out_dir    = os.path.join(self.inter_data_dir, 'nn_norm_mgc_lf0_vuv_bap_187')

        impossible_int = int(-99999)
        impossible_int = int(-99999)
        impossible_float = float(-99999.0)

        user_options = [

            # General paths
            ('work_dir', self.work_dir, 'Paths', 'work'),
            ('data_dir', self.data_dir, 'Paths', 'data'),
            ('plot_dir', self.model_dir, 'Paths', 'plot'),
            ('model_dir', self.model_dir, 'Paths', 'models'),
            ('stats_dir', self.stats_dir, 'Paths', 'stats'),
            ('gen_dir', self.gen_dir, 'Paths', 'gen'),

            # Output data paths
            ('out_feat_dir_dur', self.out_feat_dir_dur, 'Paths', 'out_feat'),
            ('out_feat_dir_cmp', self.out_feat_dir_cmp, 'Paths', 'out_feat'),
            ('out_feat_dir_dur_norm', self.out_feat_dir_dur_norm, 'Paths', 'out_feat'),
            ('out_feat_dir_cmp_norm', self.out_feat_dir_cmp_norm, 'Paths', 'out_feat'),

            # Input data paths
            ('inp_feat_dir_dur', self.inp_feat_dir_dur, 'Paths', 'inp_feat'),
            ('inp_feat_dir_cmp', self.inp_feat_dir_cmp, 'Paths', 'inp_feat'),
            ('bin_lab_dir_dur', self.bin_lab_dir_dur, 'Paths', 'inp_feat'),
            ('bin_lab_dir_cmp', self.bin_lab_dir_cmp, 'Paths', 'inp_feat'),
            ('bin_lab_dir_dur_nosilence', self.bin_lab_dir_dur_nosilence, 'Paths', 'inp_feat'),
            ('bin_lab_dir_cmp_nosilence', self.bin_lab_dir_cmp_nosilence, 'Paths', 'inp_feat'),
            ('bin_lab_dir_dur_nosilence_norm', self.bin_lab_dir_dur_nosilence_norm, 'Paths', 'inp_feat'),
            ('bin_lab_dir_cmp_nosilence_norm', self.bin_lab_dir_cmp_nosilence_norm, 'Paths', 'inp_feat'),

            # TODO: Where is the actual file list? Fix these variables -- I believe this is fixed
            ('file_id_scp', os.path.join(self.data_dir, 'processors/duration_predictor/filelist.txt'), 'Paths', 'file_id_list'),
            # ('test_id_scp', os.path.join(self.data_dir, 'test_id_list.scp'), 'Paths', 'test_id_list'),

            # Labels
            ('label_type', 'phone_align', 'Labels', 'label_type'),
            ('silence_pattern', ['*-#+*'], 'Labels', 'silence_pattern'),

            # Input-Output
            # TODO: I can ad dur to this list to combine duration and acoustic modeling
            ('output_features', ['mgc', 'lf0', 'vuv', 'bap'], 'Input-Output', 'output_features'),
            ('model_output_type', 'acoustic', 'Input-Output', 'model_output_type'),

            ('inp_dim', self.inp_dim, 'Input-Output', 'inp_dim'),
            # ('out_dim', 187, 'Input-Output', 'out_dim'),

            ('mgc_dim', 60, 'Input-Output', 'mgc'),
            ('lf0_dim', 1, 'Input-Output', 'lf0'),
            ('bap_dim', 5, 'Input-Output', 'bap'),
            ('dmgc_dim', 180, 'Input-Output', 'mgc'),
            ('dlf0_dim', 3, 'Input-Output', 'lf0'),
            ('dbap_dim', 15, 'Input-Output', 'bap'),
            ('dur_dim', 5, 'Input-Output', 'cmp'),
            ('cmp_dim', 60*3 + 1*3 + 5*3, 'Input-Output', 'cmp'),

            ('inp_file_ext', '.lab', 'Input-Output', 'inp_file_ext'),
            ('out_file_ext', '.cmp', 'Input-Output', 'out_file_ext'),

            ('mgc_ext', '.mgc', 'Input-Output', 'mgc_ext'),
            ('bap_ext', '.bap', 'Input-Output', 'bap_ext'),
            ('lf0_ext', '.lf0', 'Input-Output', 'lf0_ext'),
            ('cmp_ext', '.cmp', 'Input-Output', 'cmp_ext'),
            ('lab_ext', '.lab', 'Input-Output', 'lab_ext'),
            ('utt_ext', '.utt', 'Input-Output', 'utt_ext'),
            ('stepw_ext', '.stepw', 'Input-Output', 'stepw_ext'),
            ('sp_ext', '.sp', 'Input-Output', 'sp_ext'),
            ('dur_ext', '.dur', 'Input-Output', 'dur_ext'),

            ('inp_norm', 'MVN', 'Input-Output', 'inp_norm'),
            ('out_norm', 'MVN', 'Input-Output', 'out_norm'),

            # Architecture
            ('hidden_layer_type', ['tanh', 'tanh', 'tanh', 'tanh'], 'Architecture', 'hidden_layer_type'),
            ('hidden_layer_size', [ 1024 ,  1024 ,  1024 ,  1024 ], 'Architecture', 'hidden_layer_size'),
            ('shared_layer_flag', [0, 0, 0, 0], 'Architecture', 'shared_layer_flag'),
            ('speaker_id', ['placeholder'], 'Architecture', 'speaker_id'),

            ('batch_size'   , 256, 'Architecture', 'batch_size'),
            ('num_of_epochs',   1, 'Architecture', 'training_epochs'),
            ('stopping_patience', 10, 'Architecture', 'stopping_patience'),
            ('restore_best_weights', True, 'Architecture', 'restore_best_weights'),
            ('dropout_rate' , 0.0, 'Architecture', 'dropout_rate'),
            ('l1_reg', 0.0, 'Architecture', 'l1_reg'),
            ('l2_reg', 0.0, 'Architecture', 'l2_reg'),

            ('output_layer_type', 'linear', 'Architecture', 'output_layer_type'),
            ('optimizer'        ,   'adam', 'Architecture', 'optimizer'),
            ('loss_function'    ,    'mse', 'Architecture', 'loss_function'),

            # RNN
            ('sequential_training', False, 'Architecture', 'sequential_training'),
            ('stateful'           , False, 'Architecture', 'stateful'),
            ('use_high_batch_size', False, 'Architecture', 'use_high_batch_size'),

            ('training_algo',   1, 'Architecture', 'training_algo'),
            ('merge_size'   ,   1, 'Architecture', 'merge_size'),
            ('seq_length'   , 200, 'Architecture', 'seq_length'),
            ('bucket_range' , 100, 'Architecture', 'bucket_range'),

            ('gpu_num', 1, 'Architecture', 'gpu_num'),

            # Data
            ('shuffle_data', True, 'Data', 'shuffle_data'),

            ('train_file_number', impossible_int, 'Data','train_file_number'),
            ('valid_file_number', impossible_int, 'Data','valid_file_number'),
            ('test_file_number' , impossible_int, 'Data','test_file_number'),

            # Processes
            ('GenTestList', False, 'Processes', 'GenTestList'),

            ('NORMDATA'   , False, 'Processes', 'NORMDATA'),
            ('TRAINDNN' , False, 'Processes', 'TRAINDNN'),
            ('TESTDNN'  , False, 'Processes', 'TESTDNN'),
            ('MAKELAB', False, 'Processes', 'MAKELAB'),
            ('MAKECMP', False, 'Processes', 'MAKECMP'),

        ]

        # this uses exec(...) which is potentially dangerous since arbitrary code could be executed
        for (variable, default, section, option) in user_options:
            # default value
            value=None

            try:
                # first, look for a user-set value for this variable in the config file
                value = cfgparser.get(section,option)
                user_or_default='user'

            except (configparser.NoSectionError, configparser.NoOptionError):
                # use default value, if there is one
                if (default == None) or \
                   (default == '')   or \
                   ((type(default) == int) and (default == impossible_int)) or \
                   ((type(default) == float) and (default == impossible_float))  :
                    logger.critical('%20s has no value!' % (section+":"+option) )
                    raise Exception
                else:
                    value = default
                    user_or_default='default'

            if type(default) == str:
                exec('self.%s = "%s"'      % (variable,value))
            elif type(default) == int:
                exec('self.%s = int(%s)'   % (variable,value))
            elif type(default) == float:
                exec('self.%s = float(%s)' % (variable,value))
            elif type(default) == bool:
                exec('self.%s = bool(%s)'  % (variable,value))
            elif type(default) == list:
                exec('self.%s = list(%s)'  % (variable,value))
            elif type(default) == dict:
                exec('self.%s = dict(%s)'  % (variable,value))
            else:
                logger.critical('Variable %s has default value of unsupported type %s',variable,type(default))
                raise Exception('Internal error in configuration settings: unsupported default type')

            logger.info('%20s has %7s value %s' % (section+":"+option,user_or_default,value) )

    def complete_configuration(self):
        # to be called after reading any user-specific settings
        # because the values set here depend on those user-specific settings

        # get a logger
        logger = logging.getLogger("configuration")

        # create directories if not exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)

        if not os.path.exists(self.gen_dir):
            os.makedirs(self.gen_dir)

        # input-output normalization stat files
        self.inp_stats_file = os.path.join(self.model_dir, "input_%d_%s.norm" %(int(self.train_file_number), self.inp_norm))

        # Create output file for each speaker in dataset (only separate speakers if shared layers exist)
        self.out_stats_file_list = []
        for speaker in self.speaker_id:
            filename = os.path.join(self.model_dir, "output_%d_%s_%s.norm" %(int(self.train_file_number), self.out_norm, speaker))
            self.out_stats_file_list.append(filename)

        # define model file name
        if self.sequential_training:
            self.combined_model_arch = 'RNN'+str(self.training_algo)
        else:
            self.combined_model_arch = 'DNN'

        self.combined_model_arch += '_'+str(len(self.hidden_layer_size))
        self.combined_model_arch += '_'+'_'.join(map(str, self.hidden_layer_size))
        self.combined_model_arch += '_'+'_'.join(map(str, self.hidden_layer_type))

        self.nnets_file_name = '%s_%d_train_%d_%d_%d_model' \
                          %(self.combined_model_arch, int(self.shuffle_data),
                            self.train_file_number, self.batch_size, self.num_of_epochs)

        logger.info('model file: %s' % (self.nnets_file_name))

        # model files
        self.json_model_file = os.path.join(self.model_dir, self.nnets_file_name+'.json')
        self.h5_model_file   = os.path.join(self.model_dir, self.nnets_file_name+'.h5')
        self.model_params_file   = os.path.join(self.model_dir, self.nnets_file_name+'.pickle')


        # predicted features directory
        self.pred_feat_dir = os.path.join(self.gen_dir, self.nnets_file_name)
        if not os.path.exists(self.pred_feat_dir):
            os.makedirs(self.pred_feat_dir)

        # string.lower for some architecture values
        self.output_layer_type = self.output_layer_type.lower()
        self.optimizer         = self.optimizer.lower()
        self.loss_function     = self.loss_function.lower()
        for i in range(len(self.hidden_layer_type)):
            self.hidden_layer_type[i] = self.hidden_layer_type[i].lower()

        # set sequential training True if using LSTMs
        if 'lstm' in self.hidden_layer_type:
            self.sequential_training = True

        # set/limit batch size to 25
        if self.sequential_training and self.batch_size>50:
            if not self.use_high_batch_size:
                logger.info('reducing the batch size from %s to 25' % (self.batch_size))
                self.batch_size = 25 ## num. of sentences in this case

        # rnn params
        # self.rnn_params = {}
        # self.rnn_params['merge_size'] = self.merge_size
        # self.rnn_params['seq_length'] = self.seq_length
        # self.rnn_params['bucket_range'] = self.bucket_range
        # self.rnn_params['stateful'] = self.stateful

        # Process critical Input-Output parameters necessary for acoustic data composition
        self.cmp_dim = 0  # number of features in the NN output
        self.in_dir_dict = {}   # where the raw acoustic feature files are found
        self.file_extension_dict = {}  # dictionary of file extensions (so that the acoustic feature files can be found)
        self.out_dimension_dict = {}  # Number of dimensions in the NN output (includes delta and acc window data)
        self.in_dimension_dict = {}
        for feat in self.output_features:
            # The directory with the output features should always be the same
            if feat == 'mgc':
                self.cmp_dim += self.dmgc_dim
                self.file_extension_dict[feat] = self.mgc_ext
                self.in_dir_dict[feat] = self.out_feat_dir_cmp
                self.out_dimension_dict[feat] = self.dmgc_dim
                self.in_dimension_dict[feat] = self.mgc_dim
            elif feat == 'bap':
                self.cmp_dim += self.dbap_dim
                self.file_extension_dict[feat] = self.bap_ext
                self.in_dir_dict[feat] = self.out_feat_dir_cmp
                self.out_dimension_dict[feat] = self.dbap_dim
                self.in_dimension_dict[feat] = self.bap_dim
            elif feat == 'lf0':
                self.cmp_dim += self.dlf0_dim
                self.file_extension_dict[feat] = self.lf0_ext
                self.in_dir_dict[feat] = self.out_feat_dir_cmp
                self.out_dimension_dict[feat] = self.dlf0_dim
                self.in_dimension_dict[feat] = self.lf0_dim

            # The "voiced un-voiced" feature is added independent of any vocoder output files
            elif feat == 'vuv':
                self.cmp_dim += 1
                self.out_dimension_dict[feat] = 1
