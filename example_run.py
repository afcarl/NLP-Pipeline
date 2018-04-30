from spacy_helper import SpacyProcessor

gn_path = "/PATH/TO/GOOGLENEWS/GoogleNews-vectors-negative300.bin.gz"
textfile = "test_doc.txt"
#spacy_nlp_path = "google_news_model"
spacy_nlp_path=None
SP = SpacyProcessor(textfile, 14, gn_path, use_google_news=True, merge=False, nlp=spacy_nlp_path)

# Compute embed matrix
SP._compute_embed_matrix()