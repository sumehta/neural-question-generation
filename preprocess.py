import os
from collections import defaultdict


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
    def __init__(self, path, sent_maxlen=500, par_maxlen=2000, src_vocab_size=50000, tgt_vocab_size=28000, par_vocab_size=50000, lowercase=False):
        self.src_dictionary = Dictionary()
        self.tgt_dictionary = Dictionary()
        self.par_dictionary = Dictionary()

        self.lowercase = lowercase
        self.sent_maxlen = sent_maxlen
        self.par_maxlen = par_maxlen

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.par_vocab_size = par_vocab_size

        self.train_src_path = os.path.join(path, 'src-train.txt')
        self.train_tgt_path = os.path.join(path, 'tgt-train.txt')
        self.train_par_path = os.path.join(path, 'para-train.txt')
        self.valid_src_path = os.path.join(path, 'src-dev.txt')
        self.valid_tgt_path = os.path.join(path, 'tgt-dev.txt')
        self.valid_par_path = os.path.join(path, 'para-dev.txt')
        self.test_src_path  = os.path.join(path, 'src-test.txt')
        self.test_tgt_path  = os.path.join(path, 'tgt-test.txt')
        self.test_par_path  = os.path.join(path, 'para-test.txt')

        # make the vocabulary from training set
        self.src_dictionary = self.make_vocab(self.train_src_path, self.src_vocab_size)
        self.tgt_dictionary = self.make_vocab(self.train_tgt_path, self.tgt_vocab_size)
        self.par_dictionary = self.make_vocab(self.train_par_path, self.par_vocab_size)

        self.train_src = self.tokenize(self.train_src_path, self.src_dictionary, self.sent_maxlen)
        self.train_tgt = self.tokenize(self.train_tgt_path, self.tgt_dictionary, self.sent_maxlen)
        self.train_para = self.tokenize(self.train_par_path, self.par_dictionary, self.par_maxlen)
        self.valid_src = self.tokenize(self.valid_src_path, self.src_dictionary, self.sent_maxlen)
        self.valid_tgt = self.tokenize(self.valid_tgt_path, self.tgt_dictionary, self.sent_maxlen)
        self.valid_para = self.tokenize(self.valid_par_path, self.par_dictionary, self.par_maxlen)
        #self.train = self.tokenize(self.train_path)
        # self.test = self.tokenize(self.test_path)

    def make_vocab(self, path, vocab_size):
        assert os.path.exists(path)

        dictionary = Dictionary()
        # Add words to the dictionary
        with open(path, 'r') as f:
            for line in f:
                if self.lowercase:
                    # -1 to get rid of \n character
                    words = line[:-1].lower().split(" ")
                else:
                    words = line[:-1].split(" ")
                for word in words:
                    dictionary.add_word(word)

        # prune the vocabulary
        dictionary.prune_vocab(k=vocab_size, cnt=False)
        return dictionary

    def tokenize(self, path, dictionary, maxlen=30):
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
                if len(words) > maxlen:
                    dropped += 1
                    continue
                words = ['<sos>'] + words
                words += ['<eos>']
                # vectorize
                vocab = dictionary.word2idx
                unk_idx = vocab['<oov>']
                indices = [vocab[w] if w in vocab else unk_idx for w in words]
                lines.append(indices)

        print("Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return lines


if __name__ == '__main__':

    #TODO: Use a ConfigParser to parse data paths specified via INI config
    corpus = Corpus(path='/home/sneha/Documents/dev/neural-question-generation/data/processed/')
