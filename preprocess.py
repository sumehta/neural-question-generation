import os
from collections import defaultdict

# local function main()
#   local requiredOptions = {
#     "train_src",
#     "train_tgt",
#     "train_par",
#     "valid_src",
#     "valid_tgt",
#     "valid_par",
#     "save_data"
#   }
#
#   onmt.utils.Opt.init(opt, requiredOptions)
#
#   local data = {}
#
#   data.dicts = {}
#   data.dicts.src = initVocabulary('source', opt.train_src, opt.src_vocab,
#                                   opt.src_vocab_size, opt.features_vocabs_prefix)
#   data.dicts.tgt = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
#                                   opt.tgt_vocab_size, opt.features_vocabs_prefix)
#   data.dicts.par = initVocabulary('prgrph', opt.train_par, opt.par_vocab,
#                                   opt.par_vocab_size, opt.features_vocabs_prefix)
#
#   print('Preparing training data...')
#   data.train = {}
#   data.train.src, data.train.tgt, data.train.par = makeData(opt.train_src, opt.train_tgt, opt.train_par,
#                                             data.dicts.src, data.dicts.tgt, data.dicts.par)
#   print('')
#
#   print('Preparing validation data...')
#   data.valid = {}
#   data.valid.src, data.valid.tgt, data.valid.par = makeData(opt.valid_src, opt.valid_tgt, opt.valid_par,
#                                             data.dicts.src, data.dicts.tgt, data.dicts.par)
#   print('')
#
#   if opt.src_vocab:len() == 0 then
#     saveVocabulary('source', data.dicts.src.words, opt.save_data .. '.src.dict')
#   end
#
#   if opt.tgt_vocab:len() == 0 then
#     saveVocabulary('target', data.dicts.tgt.words, opt.save_data .. '.tgt.dict')
#   end
#
#   if opt.par_vocab:len() == 0 then
#     saveVocabulary('prgrph', data.dicts.par.words, opt.save_data .. '.par.dict')
#   end
#
#   if opt.features_vocabs_prefix:len() == 0 then
#     saveFeaturesVocabularies('source', data.dicts.src.features, opt.save_data)
#     saveFeaturesVocabularies('target', data.dicts.tgt.features, opt.save_data)
#     saveFeaturesVocabularies('prgrph', data.dicts.par.features, opt.save_data)
#   end
#
#   print('Saving data to \'' .. opt.save_data .. '-train.t7\'...')
#   torch.save(opt.save_data .. '-train.t7', data, 'binary', false)
#
# end
#
# main()


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word2idx['<pad>'] = 0
        self.word2idx['<sos>'] = 1
        self.word2idx['<eos>'] = 2
        self.word2idx['<oov>'] = 3
        self.wordcounts = defaultdict(int)

    # to track word counts
    def add_word(self, word):
        self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5, cnt=False):
        # get all words and their respective counts
        vocab_list = [(word, count) for word, count in self.wordcounts.items()]
        if cnt:
            # prune by count
            self.pruned_vocab = \
                    {pair[0]: pair[1] for pair in vocab_list if pair[1] > k}
        else:
            # prune by most frequently seen words
            vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]
        # sort to make vocabulary determistic
        self.pruned_vocab.sort()

        # add all chosen words to new vocabulary/dict
        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        print("original vocab {}; pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
      return len(self.word2idx)


class Corpus(object):
    def __init__(self, path, maxlen, vocab_size=11000, lowercase=False):
        self.dictionary = Dictionary()
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.train_path = os.path.join(path, 'train.txt')
        self.test_path = os.path.join(path, 'test.txt')
        self.valid_path = os.path.join(path, 'valid.txt')
        # make the vocabulary from training set
        self.make_vocab()

        self.train = self.tokenize(self.train_path)
        self.test = self.tokenize(self.test_path)

    def make_vocab(self):
        assert os.path.exists(self.train_path)
        # Add words to the dictionary
        with open(self.train_path, 'r') as f:
            for line in f:
                if self.lowercase:
                    # -1 to get rid of \n character
                    words = line[:-1].lower().split(" ")
                else:
                    words = line[:-1].split(" ")
                for word in words:
                    self.dictionary.add_word(word)

        # prune the vocabulary
        self.dictionary.prune_vocab(k=self.vocab_size, cnt=False)

    def tokenize(self, path):
        """Tokenizes a text file."""
        dropped = 0
        with open(path, 'r') as f:
            linecount = 0
            lines = []
            for line in f:
                linecount += 1
                if self.lowercase:
                    words = line[:-1].lower().strip().split(" ")
                else:
                    words = line[:-1].strip().split(" ")
                if len(words) > self.maxlen:
                    dropped += 1
                    continue
                words = ['<sos>'] + words
                words += ['<eos>']
                # vectorize
                vocab = self.dictionary.word2idx
                unk_idx = vocab['<oov>']
                indices = [vocab[w] if w in vocab else unk_idx for w in words]
                lines.append(indices)

        print("Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return lines