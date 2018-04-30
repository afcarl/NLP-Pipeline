import numpy as np
import spacy
from spacy.attrs import LOWER, LIKE_URL, LIKE_EMAIL, ORTH, IS_PUNCT
from spacy.pipeline import Tagger, DependencyParser, EntityRecognizer
import gensim
import tensorflow as tf
import os

class SpacyProcessor:

    def __init__(self, textfile, max_length, gn_path, use_google_news=True,
     nlp=None, skip="<SKIP>", merge=False, num_threads=8):
        """Summary

        Args:
            textfile (TYPE): Description
            max_length (TYPE): Description
            skip (TYPE, optional): Short documents will be padded with this variable up until max_length
            gn_path (TYPE): Description
            use_google_news (bool, optional): Description
            nlp (None, optional): Description
        """
        self.gn_path = gn_path
        self.textfile = textfile
        self.max_length = max_length
        self.skip = skip
        self.nlp = nlp
        self.merge = merge
        self.num_threads = num_threads
        if self.nlp == None:
            if use_google_news:
                self.load_google_news()
            else:
                self.nlp = spacy.load('en_core_web_lg')
        else:
            if os.path.exists(self.nlp):
                self.nlp = spacy.load(self.nlp)
            elif use_google_news:
                self.load_google_news()
            else:
                self.nlp = spacy.load('en_core_web_lg')
        self.tokenize()

    def load_google_news(self):
        '''
        to get frequencies
        vocab_obj = model.vocab["word"]
        vocab_obj.count
        '''

        # Load google news vecs in gensim
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.gn_path, binary=True)

        # Init blank english spacy nlp object
        self.nlp = spacy.load('en_core_web_lg', vectors=False)

        # Loop through range of all indexes, get words associated with each index.
        # The words in the keys list will correspond to the order of the google embed matrix
        self.keys = []
        for idx in range(3000000):
            word = self.model.index2word[idx]
            word = word.lower()
            self.keys.append(word)
            # Add the word to the nlp vocab
            self.nlp.vocab.strings.add(word)


        # Set the vectors for our nlp object to the google news vectors
        self.nlp.vocab.vectors = spacy.vocab.Vectors(data=self.model.syn0, keys=self.keys)

        # We have to manually add these tools when initializing nlp object as blank
        #T  = Tagger(self.nlp.vocab)
        #D = DependencyParser(self.nlp.vocab)
        #E = EntityRecognizer(self.nlp.vocab)

        # pipes = ["tagger", "parser", "ner"]
        # for name in pipes:
        #     component = self.nlp.create_pipe(name)
        #     self.nlp.add_pipe(component)

    def tokenize(self):
        # Read in text data from textfile path
        self.texts = open(self.textfile).read().split('\n')
        # Init data
        self.data = np.zeros((len(self.texts), self.max_length), dtype=np.uint64)
        
        # Add the skip token to the vocab, creating a unique hash for it
        self.nlp.vocab.strings.add(self.skip)
        self.skip = self.nlp.vocab.strings[self.skip]
        self.data[:] = self.skip

        self.bad_deps = ("amod", "compound", "punct")

        # REMOVE THIS...It should be at the doc.to_array line where LOWER is
        attr = LOWER
        for row, doc in enumerate(self.nlp.pipe(self.texts, n_threads=self.num_threads, batch_size=10000)):
            self.doc=doc
            if self.merge:
                # from the spacy blog, an example on how to merge
                # noun phrases into single tokens
                for phrase in list(doc.noun_chunks):

                    while len(phrase) > 1 and phrase[0].dep_ not in self.bad_deps:
                        phrase = phrase[1:]
                    if len(phrase) > 1:
                        # merge the tokens, e.g. good_ideas
                        phrase.merge(tag=phrase.root.tag_, lemma=phrase.text,
                                     ent_type=phrase.root.ent_type_)

                        #TODO DELETE THIS IT IS FOR DEBUGGING
                        self.phrase = phrase
                    # iterate over named entities
                    for ent in doc.ents:
                        if len(ent) > 1:
                            # Merge them into single tokens.
                            ent.merge(tag=ent.root.tag_, lemma=ent.text, ent_type=ent.label_)

            dat = doc.to_array([LOWER, LIKE_EMAIL, LIKE_URL, IS_PUNCT])
            #dat = doc.to_array([ORTH, LIKE_EMAIL, LIKE_URL]).astype('int32')
            self.dat = dat
            if len(dat) > 0:
                ##            dat = dat.astype('int32')
                #msg = "Negative indices reserved for special tokens"
                #assert dat.min() >= 0, msg
                # Replace email and URL tokens
                idx = (dat[:, 1] > 0) | (dat[:, 2] > 0)
                dat[idx] = self.skip
                # Delete punctuation
                delete = np.where(dat[:,3]==1)
                dat = np.delete(dat, delete, 0)
                length = min(len(dat), self.max_length)
                self.data[row, :length] = dat[:length, 0].ravel()
        self.uniques = np.unique(self.data)
        self.vocab = self.nlp.vocab
        # Making an idx to word mapping for vocab
        self.idx_to_word = {}
        # Manually putting in this hash for the padding ID
        self.idx_to_word[self.skip] = '<SKIP>'
        
        for v in self.uniques:
            if v!= self.skip:
                try:
                    self.idx_to_word[v] = self.nlp.vocab[v].lower_
                except:
                    pass        
        #self.idx_to_word = {v: self.nlp.vocab[v].lower_ for v in self.uniques if v != self.skip}
    def _compute_embed_matrix(self):
        #Returns list of values and their frequencies
        self.unique, self.freqs= np.unique(self.data, return_counts=True)

        ##Sort unique hash id values by frequency
        self.hash_ids = [x for _,x in sorted(zip(self.freqs, self.unique), reverse=True)]
        self.freqs = sorted(self.freqs, reverse=True)

        ##Create word id's starting at 0
        self.word_ids = np.arange(len(self.hash_ids))

        self.hash2idx = dict(zip(self.hash_ids, self.word_ids))
        self.idx2hash = dict(zip(self.word_ids, self.hash_ids))

        ## Create embed matrix
        zeros = np.zeros(300)
        embed_matrix = []

        for i, h in enumerate(self.hash_ids):
            vector = self.nlp.vocab[h].vector
            if np.array_equal(zeros, vector):
                # If oov, init a random uniform vector
                vector = np.random.uniform(-1,1,300)
                embed_matrix.append(vector)
            else:
                embed_matrix.append(vector)
        self.embed_matrix = np.array(embed_matrix)

        self.embed_matrix_tensor = tf.convert_to_tensor(self.embed_matrix)
        self.embed_matrix_var = tf.Variable(self.embed_matrix_tensor)

    def save_nlp_object(self, nlp_object_path):
        self.nlp.to_disk(nlp_object_path)

    def idx_seq_to_words(self, seq):
        '''
        Pass this a single tokenized list of hash IDs and it will
        translate it to words!
        '''
        words = " "
        words = words.join([self.idx_to_word[seq[i]] for i in range(seq.shape[0])])
        return words


#SP = SpacyProcessor(textfile, gn_path)
