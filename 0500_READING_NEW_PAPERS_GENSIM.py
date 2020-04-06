import pandas as pd
import numpy as np

df_1 = pd.read_csv('data/articles1.csv')

df_2 = pd.read_csv('data/articles2.csv')
df_2.head()

# #### let's select the first 50 new papers

new_titles = df_2['title'][:50].array
new_papers = df_2['content'][:50].array

new_titles[34]

# +
import pickle
import nltk
import gensim

dictionary = gensim.corpora.Dictionary.load('models/dictionary.gensim')

with open("lists/bow_corpus.txt", "rb") as fp:   # Unpickling
    bow_corpus = pickle.load(fp)

with open("lists/norm_corpus_bigrams.txt", "rb") as fp: 
    norm_corpus_bigrams = pickle.load(fp)

with open("lists/norm_papers.txt", "rb") as fp:
    norm_papers = pickle.load(fp)

with open("lists/pre_papers.txt", "rb") as fp:   
    pre_papers = pickle.load(fp)

with open("lists/pre_titles.txt", "rb") as fp:  
    pre_titles = pickle.load(fp)
# -

# ### PREPROCESS NEW PAPERS

# first preprcoess these new papers and extract features using the same sequence of steps we followed when building the topic models.
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

# we have pre_papers and pre_titles because the normalizing function removes empty papers and titles
# so for consistency the papers and titles that we perform LDA on will be kept 


# -

# #### let's create a text wrangling and feature engineering pipeline, which should match the same steps we followed when training our topic model.
#

bigram_model = gensim.models.phrases.Phraser.load('models/bigram_model.gensim')


# +
def text_preprocessing_pipeline(documents, normaliser_fn, bigram_model, titles):
    norm_docs, pre_papers, pre_titles = normaliser_fn(documents, titles)
    norm_docs_bigrams = bigram_model[norm_docs]
    return norm_docs_bigrams, pre_papers, pre_titles

def bow_features_pipeline(tokenized_docs, dictionary):
    paper_bow_features = [dictionary.doc2bow(text)
                              for text in tokenized_docs]
    return paper_bow_features

norm_new_papers, new_pre_papers, new_pre_titles = text_preprocessing_pipeline(documents=new_papers,
                                                                    normaliser_fn=normalise_corpus,
                                                                    bigram_model=bigram_model, 
                                                                    titles=new_titles)

norm_bow_features = bow_features_pipeline(tokenized_docs=norm_new_papers,
                                         dictionary=dictionary)
# -

print(norm_new_papers[0][:30])

print(norm_bow_features[0][:30])

# ### LOAD LDA MODEL

# +
TOPICS = 25

load_lda_model = gensim.models.ldamodel.LdaModel.load('models/gensim/model_'+str(TOPICS)+'.gensim')

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

# ### PREDICT NEW TOPICS OF PAPERS

def get_topic_predictions(topic_model, corpus, topn=3):
    topic_predictions = topic_model[corpus]
    best_topics = [[(topic, round(wt, 3)) 
                        for topic, wt in sorted(topic_predictions[i], 
                                                key=lambda row: -row[1])[:topn]] 
                            for i in range(len(topic_predictions))]
    return best_topics


topic_preds = get_topic_predictions(topic_model=load_lda_model, 
                                    corpus=norm_bow_features, topn=2)

# #### building a results df

# +
results_df = pd.DataFrame()
results_df['Papers'] = range(1, len(new_pre_papers)+1)
results_df['Dominant Topics'] = [[topic_num+1 for topic_num, wt in item] for item in topic_preds]
res = results_df.set_index(['Papers'])['Dominant Topics'].apply(pd.Series).stack().reset_index(level=1, drop=True)
results_df = pd.DataFrame({'Dominant Topics': res.values}, index=res.index)
results_df['Contribution %'] = [topic_wt for topic_list in 
                                        [[round(wt*100, 2) 
                                              for topic_num, wt in item] 
                                                 for item in topic_preds] 
                                    for topic_wt in topic_list]

results_df['Topic Desc'] = [topics_df.iloc[t-1]['Terms per Topic'] for t in results_df['Dominant Topics'].values]
results_df['Title'] = [new_pre_titles[i-1][:200] for i in results_df.index.values]
results_df['Paper Desc'] = [new_pre_papers[i-1][:200] for i in results_df.index.values]

# +
pd.set_option('display.max_colwidth', 300)

results_df.sort_values(by='Contribution %', ascending=False)
# -

# Looking at the generated topics for the new, previously unseen papers, I would say our model has done an excellent job!











# ### PREDICTING WITH MALLET

load_lda_model

# +
TOPICS = 25

load_lda_model = gensim.models.wrappers.LdaMallet.load('models/mallet/model_'+str(TOPICS)+'.gensim')


# convert the ldaMallet to LdaModel. It was the only way to get some result with loading mallet model.
load_lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(load_lda_model)

topics = [[(term, round(wt, 3))
               for term, wt in load_lda_model.show_topic(n, topn=20)]
                   for n in range(0, load_lda_model.num_topics)]

pd.set_option('display.max_colwidth', -1)

topics_df = pd.DataFrame([', '.join([term for term, wt in topic])
                              for topic in topics],
                         columns = ['Terms per Topic'],
                         index=['Topic'+str(t) for t in range(1, load_lda_model.num_topics+1)]
                         )


# -

def get_topic_predictions(topic_model, corpus, topn=3):
    topic_predictions = topic_model[corpus]
    best_topics = [[(topic, round(wt, 3)) 
                        for topic, wt in sorted(topic_predictions[i], 
                                                key=lambda row: -row[1])[:topn]] 
                            for i in range(len(topic_predictions))]
    return best_topics


topic_preds = get_topic_predictions(topic_model=load_lda_model, 
                                    corpus=norm_bow_features, topn=2)

# +
results_df = pd.DataFrame()
results_df['Papers'] = range(1, len(new_pre_papers)+1)
results_df['Dominant Topics'] = [[topic_num+1 for topic_num, wt in item] for item in topic_preds]
res = results_df.set_index(['Papers'])['Dominant Topics'].apply(pd.Series).stack().reset_index(level=1, drop=True)
results_df = pd.DataFrame({'Dominant Topics': res.values}, index=res.index)
results_df['Contribution %'] = [topic_wt for topic_list in 
                                        [[round(wt*100, 2) 
                                              for topic_num, wt in item] 
                                                 for item in topic_preds] 
                                    for topic_wt in topic_list]

results_df['Topic Desc'] = [topics_df.iloc[t-1]['Terms per Topic'] for t in results_df['Dominant Topics'].values]
results_df['Title'] = [new_pre_titles[i-1][:200] for i in results_df.index.values]
results_df['Paper Desc'] = [new_pre_papers[i-1][:200] for i in results_df.index.values]


pd.set_option('display.max_colwidth', 300)

results_df.sort_values(by='Contribution %', ascending=False)
# -


