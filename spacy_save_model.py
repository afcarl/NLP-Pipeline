import spacy
from gensim.models import KeyedVectors

'''
Testing to see if the accepted post works from:
https://stackoverflow.com/questions/42094180/spacy-how-to-load-google-news-word2vec-vectors
'''
gn_path = "/path/to/googlenews/vecs/GoogleNewsVectors/GoogleNews-vectors-negative300.bin.gz"


# Run this first
def load_and_save_vectors(gn_path):
    model = KeyedVectors.load_word2vec_format(gn_path, binary=True)
    model.wv.save_word2vec_format('googlenews.txt')


'''
Then:

1) Remove the first line of the .txt

tail -n +2 googlenews.txt > googlenews.new && mv -f googlenews.new googlenews.txt

2) Compress the txt as bz2

bzip2 googlenews.txt

'''


# Next, Run this on the zipped file
def create_spacy_bin_file():
    spacy.vocab.write_binary_vectors('googlenews.txt.bz2', 'googlenews.bin')


# Move the output file from the above function to site-packages or dist-packages: (You will have to create some directories)
'''/lib/python/site-packages/spacy/data/en_google-1.0.0/vocab/googlenews.bin'''


# Now we should be able to load the vectors:
def load_vecs_live():
    # note, this would have to run in the same session
    nlp = spacy.load('en', vectors="en_google")


# because I dont have the symbolic link, mine looks like this:
# nlp = spacy.load('en_core_web_lg', vectors="en_google")

def load_vecs():
    nlp = spacy.load('en')
    nlp.vocab.load_vectors_from_bin_loc('googlenews.bin')
    return nlp
