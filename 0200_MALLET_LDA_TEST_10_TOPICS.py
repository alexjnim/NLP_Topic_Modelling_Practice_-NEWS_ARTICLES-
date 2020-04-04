# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np

df_1 = pd.read_csv('data/articles1.csv')
df_1

# ### LOAD DICTIONARY AND ALL LISTS

# +
import pickle

with open("lists/bow_corpus.txt", "rb") as fp:   # Unpickling
    bow_corpus = pickle.load(fp)

with open("lists/norm_corpus_bigrams.txt", "rb") as fp:   # Unpickling
    norm_corpus_bigrams = pickle.load(fp)

with open("lists/norm_papers.txt", "rb") as fp:   # Unpickling
    norm_papers = pickle.load(fp)

with open("lists/pre_papers.txt", "rb") as fp:   # Unpickling
    pre_papers = pickle.load(fp)

with open("lists/pre_titles.txt", "rb") as fp:   # Unpickling
    pre_titles = pickle.load(fp)

# +
import nltk
import gensim

dictionary = gensim.corpora.Dictionary.load('models/dictionary.gensim')

# -

# ### LOADING MALLET
#
# The MALLET framework is a Java-based package for statistical natural language processing, document classification, clustering, topic modeling, information extraction, and other machine learning applications to text. MALLET stands for MAchine Learning for LanguagE Toolkit. It was developed by Andrew McCallum along with several people at the University of Massachusetts Amherst. The MALLET topic modeling toolkit contains efficient, sampling-based implementations of Latent Dirichlet Allocation, Pachinko Allocation, and Hierarchical LDA. To use MALLET’s capabilities, we need to download the framework.

# +
### loading MALLET

# #!wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
    
# #!unzip -q mallet-2.0.8.zip

MALLET_PATH = 'mallet-2.0.8/bin/mallet'
# -

# ### RUNNING MALLET

# +
# %%time

TOTAL_TOPICS = 10

MALLET_PATH = 'mallet-2.0.8/bin/mallet'
lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path=MALLET_PATH, corpus=bow_corpus,
                                              num_topics=TOTAL_TOPICS, id2word=dictionary,
                                              iterations=500, workers=16)

# +
### save model

lda_mallet.save('models/mallet/model_'+str(TOTAL_TOPICS)+'.gensim')
# -

# ### CHECKING TOPICS

topics = [[(term, round(wt, 3))
               for term, wt in lda_mallet.show_topic(n, topn=20)]
                   for n in range(0, TOTAL_TOPICS)]
for idx, topic in enumerate(topics):
    print('Topic #'+str(idx+1)+':')
    print([term for term, wt in topic])
    print()

topics_df = pd.DataFrame([[term for term, wt in topic]
                              for topic in topics],
                         columns = ['Term'+str(i) for i in range(1, 21)],
                         index=['Topic '+str(t) for t in range(1, lda_mallet.num_topics+1)]).T
topics_df

pd.set_option('display.max_colwidth', -1)
topics_df = pd.DataFrame([', '.join([term for term, wt in topic])
                              for topic in topics],
                         columns = ['Terms per Topic'],
                         index=['Topic'+str(t) for t in range(1, lda_mallet.num_topics+1)]
                         )
topics_df

# ### EVALUATING MODEL 


cv_coherence_model_lda_mallet = gensim.models.CoherenceModel(model=lda_mallet,
                                              corpus=bow_corpus,
                                              texts=norm_corpus_bigrams,
                                              dictionary=dictionary,
                                              coherence='c_v')
avg_coherence_cv = cv_coherence_model_lda_mallet.get_coherence()
umass_coherence_model_lda_mallet = gensim.models.CoherenceModel(model=lda_mallet,
                                              corpus=bow_corpus,
                                              texts=norm_corpus_bigrams,
                                              dictionary=dictionary,
                                              coherence='u_mass')
avg_coherence_umass = umass_coherence_model_lda_mallet.get_coherence()

perplexity = -8.53533
print('Avg. Coherence Score (Cv):', avg_coherence_cv)
print('Avg. Coherence Score (UMass):', avg_coherence_umass)
print('Model Perplexity:', perplexity)


