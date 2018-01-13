import sys
import json


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import to_gpu, Corpus, batchify

parser = argparse.ArgumentParser(description='Pytoch nqg')
parser.add_argument('-data_path', type=str, required=True)
parser.add_argument('-save_model', type=str, default='', description='Model filename (the model will be saved <save_model>_epochN_PPL.t7 where PPL is the validation perplexity')
parser.add_argument('-train_from', type=str, description='If training from a checkpoint then this is the path to the pretrained model.')
parser.add_argument('-continue', action='store_true', description = 'If training from a checkpoint, whether to continue the training in the same configuration or not.')


# "**Model options**"


parser.add_argument('-layers', type=int, default=2, description='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-para_rnn_size', type=int, default=250, description='Size of Paragraph LSTM hidden states')
parser.add_argument('-sent_rnn_size', type=int, default=250, description='Size of Sentence LSTM hidden states')
parser.add_argument('-rnn_size', default=500, description='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', default=500, description='Word embedding sizes')
parser.add_argument('-feat_merge', 'concat', description='Merge action for the features embeddings: concat or sum')
parser.add_argument('-feat_vec_exponent', default=0.7, description='When using concatenation, if the feature takes N values '
                                                                   'then the embedding dimension will be set to N^exponent')
parser.add_argument('-feat_vec_size', default=20, description='When using sum, the common embedding size of the features')
parser.add_argument('-input_feed', default=1, description='Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.')
parser.add_argument('-residual', action='store_true', description='Add residual connections between RNN layers.')
parser.add_argument('-brnn', action='store_true', description='Use a bidirectional encoder')
# parser.add_argument('-brnn_merge', 'sum', [[Merge action for the bidirectional hidden states: concat or sum]])
parser.add_argument('-para_sent_encoder', action='store_false', description='Use another paragraph-level encoder')

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

parser.add_argument('-max_batch_size', type=int, default=64, description='Maximum batch size')
parser.add_argument('-epochs', type=int, default=13, description='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1, description='If loading from a checkpoint, the epoch from which to start')
parser.add_argument('-start_iteration', type=int, default=1, description='If loading from a checkpoint, the iteration from which to start')
parser.add_argument('-param_init', type=float, default=0.1, description='Parameters are initialized over uniform distribution with support (-param_init, param_init)')
parser.add_argument('-optim', 'sgd', description='Optimization method. Possible options are: sgd, adagrad, adadelta, adam')
parser.add_argument('-learning_rate', type=int, default=1, description='Starting learning rate. If adagrad/adadelta/adam is used,'
                                                                       'then this is the global learning rate. Recommended settings are: sgd = 1, '
                                                                       'adagrad = 0.1, adadelta = 1, adam = 0.0002')
parser.add_argument('-max_grad_norm', type=int, default=5, description='If the norm of the gradient vector exceeds this renormalize it '
                                                                       'to have the norm equal to max_grad_norm')
parser.add_argument('-dropout', type=float, default=0.3, description='Dropout probability. Dropout is applied between vertical LSTM stacks.')
parser.add_argument('-learning_rate_decay', type=float, default=0.5, description='Decay learning rate by this much if (i) perplexity does not'
                                                                                 ' decrease on the validation set or (ii) epoch has gone past'
                                                                                 ' the start_decay_at_limit')
parser.add_argument('-start_decay_at', type=int, default=9, description='Start decay after this epoch')
parser.add_argument('-curriculum', type=int, default=0, description='For this many epochs, order the minibatches based on source '
                                                                    'sequence length. Sometimes setting this to 1 will increase convergence speed.')
parser.add_argument('-pre_word_vecs_sent_enc', '', description='If a valid path is specified, then this will load pretrained word embeddings on the sentence encoder side '
                                                               'See README for specific formatting instructions.')
parser.add_argument('-pre_word_vecs_para_enc', '', description='If a valid path is specified, then this will load pretrained word embeddings'
                                                               ' on the paragraph encoder side. '
                                                               'See README for specific formatting instructions.')
parser.add_argument('-pre_word_vecs_dec', '', description='If a valid path is specified, then this will load '
                                                          'pretrained word embeddings on the decoder side. '
                                                          'See README for specific formatting instructions.')
parser.add_argument('-fix_word_vecs_sent_enc', action='store_false', description='Fix word embeddings on the sentence encoder side')
parser.add_argument('-fix_word_vecs_para_enc', action='store_false', description='Fix word embeddings on the paragraph encoder side')
parser.add_argument('-fix_word_vecs_dec'     , action='store_false', description='Fix word embeddings on the decoder side')



# -- GPU
parser.add_argument('-gpuid', default=0, description='1-based identifier of the GPU to use. CPU is used when the option is < 1')
parser.add_argument('-nparallel', default=1, description='When using GPUs, how many batches to execute in parallel. '
                                                         'Note: this will technically change the final batch size to max_batch_size*nparallel.')
parser.add_argument('-async_parallel', action='store_false', description='Use asynchronous parallelism training.')
parser.add_argument('-async_parallel_minbatch', default=1000, description='For async parallel computing, minimal number of batches before '
                                                                          'being parallel.')
parser.add_argument('-no_nccl', action='store_false', description='Disable usage of nccl in parallel mode.')
parser.add_argument('-disable_mem_optimization', action='store_false', description='Disable sharing internal of internal buffers between clones -'
                                                                                   'which is in general safe, except if you want to look'
                                                                                   ' inside clones for visualization purpose for instance.')

# -- bookkeeping
parser.add_argument('-save_every', default=0, description='Save intermediate models every this many iterations within an epoch. '
                                                          'If = 0, will not save models within an epoch.')
parser.add_argument('-report_every', default=50, description='Print stats every this many iterations within an epoch.')
parser.add_argument('-seed', default=3435, description='Seed for random initialization')
parser.add_argument('-json_log', action='store_false',description='Outputs logs in JSON format.')
