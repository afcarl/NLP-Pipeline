from spacy_helper import SpacyProcessor
import numpy as np
import tensorflow as tf

# File where our text is stored
textfile = "test_doc.txt"

# Location of saved spacy nlp path
# TODO - currently this is ambiguous and will load the saved google news model made in spacy_save_model
spacy_nlp_path = "google_news_model"

# Initialize spacy processor
SP = SpacyProcessor(textfile, 140, use_google_news=True, merge=True, nlp=spacy_nlp_path, token_type="lower")

# Computes embedding matrix for you
SP._compute_embed_matrix(random=True)

# Converts hashes from spacy to embedding matrix indexes (stored in SP.idx_data)
SP.convert_data_to_word2vec_indexes()

# Examples of data that could be passed to write data to tfrecords
#context = np.arange(7)
#labels = np.array([[0,1], [0,1], [0,1], [1,0], [1,0], [1,0], [1,0]])

# Writes data to tfrecords file with gzip compression (Not inferred, it is a parameter)
SP.write_data_to_tfrecords("mydata.tfrecords.gzip")#Additional param options: context = context, labels=labels
