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

df_1 = df_1[df_1['content'].notna()]
df_1 = df_1.reset_index(drop=True)

titles = df_1['title'].array
papers = df_1['content'].array

titles[1]

papers[1]

# ### TEXT WRANGLING (NORMALIZING TEXT)

# We perform some basic text wrangling or preprocessing before diving into topic modeling. We keep things simple here and perform tokenization, lemmatizing nouns, and removing stopwords and any terms having a single character.
#

# +
# %%time
import nltk

stop_words = nltk.corpus.stopwords.words('english')
wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
wnl = nltk.stem.wordnet.WordNetLemmatizer()

def normalise_corpus(papers, full_df):
    norm_papers = []
    pre_papers = []
    drop_index = []
    for i in range(len(papers)):
        paper = papers[i]

        paper = paper.lower()
        paper_tokens = [token.strip() for token in wtk.tokenize(paper)]
        paper_tokens = [wnl.lemmatize(token) for token in paper_tokens if not token.isnumeric()]
        paper_tokens = [token for token in paper_tokens if len(token) > 1]
        paper_tokens = [token for token in paper_tokens if token not in stop_words]
        paper_tokens = list(filter(None, paper_tokens))

        if paper_tokens:
            norm_papers.append(paper_tokens)
            pre_papers.append(paper)
        else:
            drop_index.append(i)

    pre_df = full_df.drop(full_df.index[drop_index])
    
    return norm_papers, pre_papers, pre_df

# we have pre_papers and pre_titles because the normalizing function removes empty papers and titles
# so for consistency the papers and titles that we perform LDA on will be kept 


# +
# %%time
import nltk

stop_words = nltk.corpus.stopwords.words('english')
wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
wnl = nltk.stem.wordnet.WordNetLemmatizer()

def normalise_corpus(papers, titles):
    norm_papers = []
    pre_papers = []
    pre_titles = []
    for i in range(len(papers)):
        paper = papers[i]
        title = titles[i]

        paper = paper.lower()
        paper_tokens = [token.strip() for token in wtk.tokenize(paper)]
        paper_tokens = [wnl.lemmatize(token) for token in paper_tokens if not token.isnumeric()]
        paper_tokens = [token for token in paper_tokens if len(token) > 1]
        paper_tokens = [token for token in paper_tokens if token not in stop_words]
        paper_tokens = list(filter(None, paper_tokens))

        if paper_tokens:
            norm_papers.append(paper_tokens)
            pre_papers.append(paper)
            pre_titles.append(title)

    return norm_papers, pre_papers, pre_titles

# we have pre_papers and pre_titles because the normalizing function removes empty papers and titles
# so for consistency the papers and titles that we perform LDA on will be kept 

norm_papers, pre_papers, pre_titles = normalise_corpus(papers, titles)

len(norm_papers), len(pre_papers), len(pre_titles)
# -

len(pre_papers), len(pre_titles)

pre_df.to_csv('data/pre_df.csv')

# ### TEXT REPRESENTATION WITH FEATURE ENGINEERING
# Before feature engineering and vectorization, we want to extract some useful bi-gram based phrases from our research papers and remove some unnecessary terms. We leverage the very useful gensim.models.Phrases class for this. This capability helps us automatically detect common phrases from a stream of sentences, which are typically multi-word expressions/word n-grams. This implementation draws inspiration from the famous paper by Mikolov, et al., “Distributed Representations of Words and Phrases and their Compositionality,” which you can check out at https://arxiv.org/abs/1310.4546 . We start by extracting and generating words and bi-grams as phrases for each tokenized research paper. We can build this phrase generation model easily with the following code and test it on a sample paper.
#

# +
import gensim

# higher threshold fewer phrases
bigram = gensim.models.Phrases(norm_papers, min_count=20, threshold=20, delimiter=b'_')

bigram_model = gensim.models.phrases.Phraser(bigram)

# sample demonstration

print(bigram_model[norm_papers[0]][:50])
# -

# We can clearly see that we have single words as well as bi-grams (two words separated by an underscore), which tells us that our model works. We leverage the min_count parameter , which tells us that our model ignores all words and bi-grams with total collected count lower than 20 across the corpus (of the input paper as a list of tokenized sentences). 
#
#
# Let’s generate phrases for all our tokenized research papers and build a vocabulary that will help us obtain a unique term/phrase to number mapping (since machine or deep learning only works on numeric tensors).

# +
norm_corpus_bigrams = [bigram_model[doc] for doc in norm_papers]

# Create a dictionary representation of the documents.
dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)

print('Sample word to number mappings:', list(dictionary.items())[:15])
print('\n Total Vocabulary Size:', len(dictionary))
# -

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.6)
print('Total Vocabulary Size:', len(dictionary))

# Transforming corpus into bag of words vectors
bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]
print(bow_corpus[1][:50])

# viewing actual terms and their counts
print([(dictionary[idx] , freq) for idx, freq in bow_corpus[1][:50]])

# ### save dictionary, norm_corpus_bigrams, pre_titles, pre_papers, etc

import pickle  

# +
with open("lists/norm_corpus_bigrams.txt", "wb") as fp:
    pickle.dump(norm_corpus_bigrams, fp)  

with open("lists/pre_titles.txt", "wb") as fp:
    pickle.dump(pre_titles, fp)    
    
with open("lists/pre_papers.txt", "wb") as fp:
    pickle.dump(pre_papers, fp)    
    
with open("lists/bow_corpus.txt", "wb") as fp:
    pickle.dump(bow_corpus, fp)  

with open("lists/norm_papers.txt", "wb") as fp:
    pickle.dump(norm_papers, fp)  
# -

dictionary.save('models/dictionary.gensim')

bigram_model

bigram_model.save('models/bigram_model.gensim')


