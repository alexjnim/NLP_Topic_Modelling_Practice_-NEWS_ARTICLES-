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

# ### LDA TUNING; FINDING THE OPTIMAL NUMBER OF TOPICS (SAVE ALL MODELS & COHERENCE SCORES)
#
# Finding the optimal number of topics in a topic model is tough, given that it is like a model hyperparameter that you always have to set before training the model. We can use an iterative approach and build several models with differing numbers of topics and select the one that has the highest coherence score. To implement this method, we build the following function.

from tqdm import tqdm
def topic_model_coherence_generator(corpus, texts, dictionary,
                          start_topic_count=2, end_topic_count=10, step=1,
                                    cpus=1):
    models = []
    coherence_scores = []
    for topic_nums in tqdm(range(start_topic_count, end_topic_count+1, step)):
        gensim_lda_model = gensim.models.LdaModel(
                                              corpus=corpus,
                                              num_topics=topic_nums,
                                              id2word=dictionary,
                                              chunksize=1740,
                                              iterations=500,
                                              alpha="auto",
                                              eta="auto",
                                              passes=20
                                                )

        cv_coherence_model_gensim_lda = gensim.models.CoherenceModel(model=gensim_lda_model,
                                                    corpus=corpus,
                                                    texts=texts,
                                                    dictionary=dictionary,
                                                    coherence='c_v')

        coherence_score = cv_coherence_model_gensim_lda.get_coherence()
        coherence_scores.append(coherence_score)
        models.append(gensim_lda_model)
        
        ### saving each model
        gensim_lda_model.save('models/gensim/model_'+str(topic_nums)+'.gensim')

    return models, coherence_scores


lda_models, coherence_scores = topic_model_coherence_generator(corpus=bow_corpus,
                                                texts=norm_corpus_bigrams,
                                                dictionary=dictionary,
                                                start_topic_count=2,
                                                end_topic_count=30, step=1,
                                                cpus=16)

# +
coherence_df = pd.DataFrame({'Number of Topics': range(2, 31, 1),
                        'Coherence Score': np.round(coherence_scores, 4)})

coherence_df.sort_values(by=['Coherence Score'], ascending=False).head(10)


# +
# save coherence score df and coherence score list 
coherence_df.to_csv('models/gensim_scores/coherence_df.csv', index=False)

with open("models/gensim_scores/coherence_scores.txt", "wb") as fp:   #Pickling
    pickle.dump(coherence_scores, fp)
# -

# ### VISUALIZE COHERENCE SCORES

# +
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
# %matplotlib inline
x_ax = range(2, 31, 1)
y_ax = coherence_scores
plt.figure(figsize=(12, 6))
plt.plot(x_ax, y_ax, c="r")
plt.axhline(y=0.535, c="k", linestyle="--", linewidth=2)
plt.rcParams['figure.facecolor'] = 'white'
xl = plt.xlabel('Number of Topics')
yl = plt.ylabel('Coherence Score')
# -


