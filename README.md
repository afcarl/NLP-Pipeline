# NLP Pipeline

The spacy helper file holds a class called SpacyProcessor. This class
aims to make using SpaCy a little bit easier. Instead of having to
rewrite a large amount of SpaCy code every time you want to process text,
the idea is that you could just use this!

## Usage
### Google News Vectors
If you want to use google news vectors, follow the directions found in
***spacy_save_model.py*** to save google news vectors to a usable format.
In the future, I will have multiple options to do this.

The returned data from this function is a 2d array of SpaCy hashes.

### Running the Processor
```python
# Initialize Spacy Processor
SP = SpacyProcessor(textfile, 140, . . .)

data = SP.data

'''
example data:
array([[ 5097672513440128799, 12579384389446384672,  7425985699627899538,
        16857069738040043275,  2283656566040971221, 12513610393978129441,
        ...
        ]])
'''

```


### Additional Features

#### Embedding Matrix Computation
This class can also compute the embedding matrix for your data if you want.
You have the option to load random uniform vectors for each unique word in your
corpus, or you can use pretrained embeddings from either SpaCy or GoogleNews.
```python
# Compute Embedding Matrix --> stored in SP.embed_matrix as np array
SP._compute_embed_matrix()

# Spacy Hashes --> Embed Indexes (stored in SP.idx_data)
SP.convert_data_to_word2vec_indexes()
```

#### Save sequences to tfrecords
This function will allow you to directly save data to tfrecords for later
use in tensorflow.

***NOTE:*** Make sure you run the two lines from embed matrix computation
before trying to do this. We need to make sure we have initialized SP.idx_data
in order to run this function.
```python
SP.write_data_to_tfrecords('mydata.tfrecords.gzip')
```
