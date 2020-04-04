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

titles = df_1['title'].array
papers = df_1['content'].array

titles[1]

papers[1]

# We perform some basic text wrangling or preprocessing before diving into topic modeling. We keep things simple here and perform tokenization, lemmatizing nouns, and removing stopwords and any terms having a single character.
#

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


def normalize_corpus(papers):
    norm_papers = []
    for paper in papers:
        paper = paper.lower()
        paper_tokens = [token.strip() for token in wtk.tokenize(paper)]
        paper_tokens = [wnl.lemmatize(token) for token in paper_tokens if not token.isnumeric()]
        paper_tokens = [token for token in paper_tokens if len(token) > 1]
        paper_tokens = [token for token in paper_tokens if token not in stop_words]
        paper_tokens = list(filter(None, paper_tokens))
        if paper_tokens:
            norm_papers.append(paper_tokens)
    return norm_papers


# norm_papers = normalize_corpus(papers)

norm_papers, pre_papers, pre_titles = normalise_corpus(papers, titles)

print(len(norm_papers))
# -

len(pre_papers), len(pre_titles)

# ### topic modelling with gensim

# +
import gensim

# higher threshold fewer phrases
bigram = gensim.models.Phrases(norm_papers, min_count=20, threshold=20, delimiter=b'_')

bigram_model = gensim.models.phrases.Phraser(bigram)

# sample demonstration
print(bigram_model[norm_papers[0]][:50])

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

# ### LDA TUNING; FINDING THE OPTIMAL NUMBER OF TOPICS
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

coherence_df = pd.DataFrame({'Number of Topics': range(2, 31, 1),
                        'Coherence Score': np.round(coherence_scores, 4)})
coherence_df.sort_values(by=['Coherence Score'], ascending=False).head(10)

# ### save coherence scores and visualize 

coherence_df.to_csv('models/gensim_scores/coherence_df.csv', index=False)

with open("models/gensim_scores/coherence_scores.txt", "wb") as fp:   #Pickling
    pickle.dump(coherence_scores, fp)

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

# ### selecting best model based on coherence

best_model_idx = coherence_df[coherence_df['Number of Topics'] == 25].index[0]
best_lda_model = lda_models[best_model_idx]
best_lda_model.num_topics

# ### saving all models

### saving all models
for i in range(len(lda_models)):
    lda_models[i].save('models/gensim/model_'+
                       str(coherence_df['Number of Topics'].iloc[i])+
                       '.gensim')

# ### saving or loading best model

# +
best_lda_model.save('models/gensim/best_model'+str(best_lda_model.num_topics)+'.gensim')

# best_lda_model = gensim.models.ldamodel.LdaModel.load('models/gensim/best_model_25.gensim')
# -

# ### checking topics

topics = [[(term, round(wt, 3))
               for term, wt in best_lda_model.show_topic(n, topn=20)]
                   for n in range(0, best_lda_model.num_topics)]
for idx, topic in enumerate(topics):
    print('Topic #'+str(idx+1)+':')
    print([term for term, wt in topic])
    print()

# A better way of visualizing the topics is to build a term-topic dataframe.

topics_df = pd.DataFrame([[term for term, wt in topic]
                              for topic in topics],
                         columns = ['Term'+str(i) for i in range(1, 21)],
                         index=['Topic '+str(t) for t in range(1, best_lda_model.num_topics+1)]).T
topics_df

# Another easy way to view the topics is to create a topic-term dataframe, whereby each topic is represented in a row with the terms of the topic being represented as a comma-separated string.

pd.set_option('display.max_colwidth', -1)
topics_df = pd.DataFrame([', '.join([term for term, wt in topic])
                              for topic in topics],
                         columns = ['Terms per Topic'],
                         index=['Topic'+str(t) for t in range(1, best_lda_model.num_topics+1)]
                         )
topics_df

# ### Interpreting results

tm_results = best_lda_model[bow_corpus]

corpus_topics = [sorted(topics, key=lambda record: -record[1])[0]
                     for topics in tm_results]
corpus_topics[:5]

corpus_topic_df = pd.DataFrame()
corpus_topic_df['Document'] = range(0, len(pre_papers))
corpus_topic_df['Dominant Topic'] = [item[0]+1 for item in corpus_topics]
corpus_topic_df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]
corpus_topic_df['Topic Desc'] = [topics_df.iloc[t[0]]['Terms per Topic'] for t in corpus_topics]
corpus_topic_df['Title'] = pre_titles
corpus_topic_df['Paper'] = pre_papers


# +
pd.set_option('display.max_colwidth', 200)

topic_stats_df = corpus_topic_df.groupby('Dominant Topic').agg({
                                                'Dominant Topic': {
                                                    'Doc Count': np.size,
                                                    '% Total Docs': np.size }
                                              })
topic_stats_df = topic_stats_df['Dominant Topic'].reset_index()
topic_stats_df['% Total Docs'] = topic_stats_df['% Total Docs'].apply(lambda row: round((row*100) / len(pre_papers), 2))
topic_stats_df['Topic Desc'] = [topics_df.iloc[t]['Terms per Topic'] for t in range(len(topic_stats_df))]
topic_stats_df
# -

# ### Document's most dominant topic with highest contribution %

corpus_topic_df.sort_values(by='Contribution %', ascending=False)

# ### Dominant Topics in Specific Papers
# Another interesting perspective is to select specific papers, view the most dominant topic in each of those papers, and see if that makes sense.

pd.set_option('display.max_colwidth', 200)
(corpus_topic_df[corpus_topic_df['Document']
                 .isin([681, 9, 392, 1622, 17,
                        906, 996, 503, 13, 733])])

# Papers on reinforcement learning, signal processing, gaussian mixture models, processor simulations, word recognitions, and many more have corresponding relevant topics as the most dominant topics. This tells us that our topic model is working well.

# ### Relevant Research Papers per Topic Based on Dominance
# A better way of representation is to try to retrieve the corresponding research paper that has the highest representation for each of the 20 topics.

corpus_topic_df.groupby('Dominant Topic').apply(lambda topic_set:
                                            (topic_set.sort_values(by=['Contribution %'],
                                                   ascending=False).iloc[0]))

# Based on the paper titles and the corresponding topics depicted, they do make sense. It looks like our model has captured the relevant latent patterns and themes in our corpus.





# ### Loading model

load_lda_model = gensim.models.ldamodel.LdaModel.load('models/gensim/model_30.gensim')

load_lda_model

# +
topics = [[(term, round(wt, 3))
               for term, wt in load_lda_model.show_topic(n, topn=20)]
                   for n in range(0, load_lda_model.num_topics)]

pd.set_option('display.max_colwidth', -1)
topics_df = pd.DataFrame([', '.join([term for term, wt in topic])
                              for topic in topics],
                         columns = ['Terms per Topic'],
                         index=['Topic'+str(t) for t in range(1, load_lda_model.num_topics+1)]
                         )
topics_df
# -

tm_results = load_lda_model[bow_corpus]

corpus_topics = [sorted(topics, key=lambda record: -record[1])[0]
                     for topics in tm_results]
corpus_topics[:5]

corpus_topic_df = pd.DataFrame()
corpus_topic_df['Document'] = range(0, len(pre_papers))
corpus_topic_df['Dominant Topic'] = [item[0]+1 for item in corpus_topics]
corpus_topic_df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]
corpus_topic_df['Topic Desc'] = [topics_df.iloc[t[0]]['Terms per Topic'] for t in corpus_topics]
corpus_topic_df['Title'] = pre_titles
corpus_topic_df['Paper'] = pre_papers

# +
pd.set_option('display.max_colwidth', 200)

topic_stats_df = corpus_topic_df.groupby('Dominant Topic').agg({
                                                'Dominant Topic': {
                                                    'Doc Count': np.size,
                                                    '% Total Docs': np.size }
                                              })
topic_stats_df = topic_stats_df['Dominant Topic'].reset_index()
topic_stats_df['% Total Docs'] = topic_stats_df['% Total Docs'].apply(lambda row: round((row*100) / len(pre_papers), 2))
topic_stats_df['Topic Desc'] = [topics_df.iloc[t]['Terms per Topic'] for t in range(len(topic_stats_df))]
topic_stats_df
# -

corpus_topic_df.sort_values(by='Contribution %', ascending=False)

corpus_topic_df.groupby('Dominant Topic').apply(lambda topic_set:
                                            (topic_set.sort_values(by=['Contribution %'],
                                                   ascending=False).iloc[0]))

# ### Visualizing topics with pyLDAvis

# +
import pyLDAvis.gensim

load_lda_model = gensim.models.ldamodel.LdaModel.load('models/gensim/model_25.gensim')

lda_display = pyLDAvis.gensim.prepare(load_lda_model, bow_corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)
# -


