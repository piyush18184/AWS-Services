import pandas as pd
from rake_nltk import Rake as rknltk
import yake
from multi_rake import Rake as rkmulti
from summa import keywords
from collections import Counter
from itertools import chain
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn import cluster
import nltk
nltk.download('all')

df = pd.read_excel('')

df[['Key Phrases Rake', 'Key Phrases Yake', 'Key Phrases Multi Rake', 'Key Phrases Summa']] = None

r = rknltk()
kw_extractor = yake.KeywordExtractor(stopwords=None)
rake = rkmulti()

stop_words = list(stopwords.words('english'))
stop_words_2 = list(get_stop_words('en'))   
stop_words.extend(stop_words_2)

if len(df[''].to_list().count(('')) > 0:
    for index, row in df.iterrows():
        text = df[''][index]
        if type(text) == str and len(text) > 2:
            word_tokens = word_tokenize(text)
            filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
            text = ' '.join(filtered_sentence)
            r.extract_keywords_from_text(text)
            df['Key Phrases Rake'][index] = r.get_ranked_phrases()[0:15]
            df['Key Phrases Yake'][index] = [t[0] for t in kw_extractor.extract_keywords(text)][0:15]
            df['Key Phrases Multi Rake'][index] = [t[0] for t in rake.apply(text)][0:15]
            df['Key Phrases Summa'][index] = [t[0] for t in keywords.keywords(text, scores=True)][0:15]
        else:
            df['Key Phrases Rake'][index] = None
            df['Key Phrases Yake'][index] = None
            df['Key Phrases Multi Rake'][index] = None
            df['Key Phrases Summa'][index] = None
                
        print("Iteration: ", index + 1, " complete")

import ast

df_kph = df[df['Key Phrases'].notna()]
df_kph_rake = df[df['Key Phrases Rake'].notna()]
df_kph_yake = df[df['Key Phrases Yake'].notna()]
df_kph_multi_rake = df[df['Key Phrases Multi Rake'].notna()]
df_kph_summa = df[df['Key Phrases Summa'].notna()]
aa = list(df_kph['Key Phrases'])
aa1 = list(df_kph_rake['Key Phrases Rake'])
aa2 = list(df_kph_yake['Key Phrases Yake'])
aa3 = list(df_kph_multi_rake['Key Phrases Multi Rake'])
aa4 = list(df_kph_summa['Key Phrases Summa'])
li_kph = []
li_kph_rake = []
li_kph_yake = []
li_kph_multi_rake = []
li_kph_summa = []

for sublist in aa:
    li_kph.append(ast.literal_eval(str(sublist)))
    
for sublist in aa1:
    li_kph_rake.append(ast.literal_eval(str(sublist)))
    
for sublist in aa2:
    li_kph_yake.append(ast.literal_eval(str(sublist)))
    
for sublist in aa3:
    li_kph_multi_rake.append(ast.literal_eval(str(sublist)))
    
for sublist in aa4:
    li_kph_summa.append(ast.literal_eval(str(sublist)))
    
kyph_count = Counter(chain.from_iterable([x for x in li_kph])).most_common()
kyph_rake_count = Counter(chain.from_iterable([x for x in li_kph_rake])).most_common()
kyph_yake_count = Counter(chain.from_iterable([x for x in li_kph_yake])).most_common()
kyph_multi_rake_count = Counter(chain.from_iterable([x for x in li_kph_multi_rake])).most_common()
kyph_summa_count = Counter(chain.from_iterable([x for x in li_kph_summa])).most_common()
kyph_count_df = pd.DataFrame(kyph_count, columns=['Key Phrase', 'Count'])
kyph_rake_count_df = pd.DataFrame(kyph_rake_count, columns=['Key Phrase Rake', 'Count'])
kyph_yake_count_df = pd.DataFrame(kyph_yake_count, columns=['Key Phrase Yake', 'Count'])
kyph_multi_rake_count_df = pd.DataFrame(kyph_multi_rake_count, columns=['Key Phrase Multi Rake', 'Count'])
kyph_summa_count_df = pd.DataFrame(kyph_summa_count, columns=['Key Phrase Summa', 'Count'])


stemmer = PorterStemmer()
sw = stopwords.words('english')

def tokenizer(keyword):
    return [stemmer.stem(w) for w in keyword.split()]

kyph_count_df1 = kyph_count_df[kyph_count_df['Count']>1]
kyph_rake_count_df1 = kyph_rake_count_df[kyph_rake_count_df['Count']>1]
kyph_yake_count_df1 = kyph_yake_count_df[kyph_yake_count_df['Count']>1]
kyph_multi_rake_count_df1 = kyph_multi_rake_count_df[kyph_multi_rake_count_df['Count']>1]
kyph_summa_count_df1 = kyph_summa_count_df[kyph_summa_count_df['Count']>1]

keywords = list(kyph_count_df1['Key Phrase'])
keywords_rake = list(kyph_rake_count_df1['Key Phrase Rake'])
keywords_yake = list(kyph_yake_count_df1['Key Phrase Yake'])
keywords_multi_rake = list(kyph_multi_rake_count_df1['Key Phrase Multi Rake'])
keywords_summa = list(kyph_summa_count_df1['Key Phrase Summa'])

li = []
for i in keywords:
    li.append(str(i))
    
li_rake = []
for i in keywords_rake:
    li_rake.append(str(i))

li_yake = []
for i in keywords_yake:
    li_yake.append(str(i))
    
li_multi_rake = []
for i in keywords_multi_rake:
    li_multi_rake.append(str(i))
    
li_summa = []
for i in keywords_summa:
    li_summa.append(str(i))
    
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens

li_sw = []
for i in li:
    word_tokens = word_tokenize(i)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    li_sw.append(' '.join(filtered_sentence))
    
li_rake_sw = []
for i in li_rake:
    word_tokens = word_tokenize(i)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    li_rake_sw.append(' '.join(filtered_sentence))
    
li_yake_sw = []
for i in li_yake:
    word_tokens = word_tokenize(i)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    li_yake_sw.append(' '.join(filtered_sentence))
    
li_multi_rake_sw = []
for i in li_multi_rake:
    word_tokens = word_tokenize(i)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    li_multi_rake_sw.append(' '.join(filtered_sentence))
    
li_summa_sw = []
for i in li_summa:
    word_tokens = word_tokenize(i)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    li_summa_sw.append(' '.join(filtered_sentence))


tfidf = TfidfVectorizer(tokenizer=tokenize)

X = pd.DataFrame(tfidf.fit_transform(li_sw).toarray(),
                 index=li_sw, columns=tfidf.get_feature_names_out())
c = cluster.AffinityPropagation()
pred = c.fit_predict(X)

X_rake = pd.DataFrame(tfidf.fit_transform(li_rake_sw).toarray(),
                 index=li_rake_sw, columns=tfidf.get_feature_names_out())
c_rake = cluster.AffinityPropagation()
pred_rake = c_rake.fit_predict(X_rake)

X_yake = pd.DataFrame(tfidf.fit_transform(li_yake_sw).toarray(),
                 index=li_yake_sw, columns=tfidf.get_feature_names_out())
c_yake = cluster.AffinityPropagation()
pred_yake = c_yake.fit_predict(X_yake)

X_multi_rake = pd.DataFrame(tfidf.fit_transform(li_multi_rake_sw).toarray(),
                 index=li_multi_rake_sw, columns=tfidf.get_feature_names_out())
c_multi_rake = cluster.AffinityPropagation()
pred_multi_rake = c_multi_rake.fit_predict(X_multi_rake)

X_summa = pd.DataFrame(tfidf.fit_transform(li_summa_sw).toarray(),
                 index=li_summa_sw, columns=tfidf.get_feature_names_out())
c_summa = cluster.AffinityPropagation()
pred_summa = c_summa.fit_predict(X_summa)

aa = X.astype(bool).sum(axis=0).sort_values(ascending = False).to_frame().reset_index()
aa.columns = ['Tags', 'Count']

aa_rake = X_rake.astype(bool).sum(axis=0).sort_values(ascending = False).to_frame().reset_index()
aa_rake.columns = ['Tags', 'Count']

aa_yake = X_yake.astype(bool).sum(axis=0).sort_values(ascending = False).to_frame().reset_index()
aa_yake.columns = ['Tags', 'Count']

aa_multi_rake = X_multi_rake.astype(bool).sum(axis=0).sort_values(ascending = False).to_frame().reset_index()
aa_multi_rake.columns = ['Tags', 'Count']

aa_summa = X_summa.astype(bool).sum(axis=0).sort_values(ascending = False).to_frame().reset_index()
aa_summa.columns = ['Tags', 'Count']

import inflect 
def get_singular(plural_noun):
    p = inflect.engine()
    plural = p.singular_noun(plural_noun)
    if (plural):
        return plural
    else:
        return plural_noun
    
tag_new = list(aa['Tags'])
li= []
for i in tag_new:
    li.append(get_singular(i))
    
aa['Tags New'] = li
aa.drop(['Tags'], axis =1)
aa = aa[['Tags New', 'Count']]
bb = aa.groupby(aa['Tags New']).aggregate({'Count':'sum'}).sort_values(by='Count').reset_index()
top_10_list = list(bb.nlargest(20, ['Count'])['Tags New'])

kyph_count_df1['Tags'] = list(kyph_count_df1['Key Phrase'].str.findall('|'.join(top_10_list)))
kyph_count_df1 = kyph_count_df1.dropna(subset=['Tags'])
for index, row in kyph_count_df1.iterrows():
    kyph_count_df1['Tags'][index] = ','.join(str(item) for item in kyph_count_df1['Tags'][index])
kyph_count_df1 = kyph_count_df1[['Tags', 'Key Phrase', 'Count']]

tag_new_rake = list(aa_rake['Tags'])
li_rake= []
for i in tag_new_rake:
    li_rake.append(get_singular(i))
    
aa_rake['Tags New'] = li_rake
aa_rake.drop(['Tags'], axis =1)
aa_rake = aa_rake[['Tags New', 'Count']]
bb_rake = aa_rake.groupby(aa_rake['Tags New']).aggregate({'Count':'sum'}).sort_values(by='Count').reset_index()
top_10_list_rake = list(bb_rake.nlargest(20, ['Count'])['Tags New'])

kyph_rake_count_df1['Tags'] = list(kyph_rake_count_df1['Key Phrase Rake'].str.findall('|'.join(top_10_list_rake)))
kyph_rake_count_df1 = kyph_rake_count_df1.dropna(subset=['Tags'])
for index, row in kyph_rake_count_df1.iterrows():
    kyph_rake_count_df1['Tags'][index] = ','.join(str(item) for item in kyph_rake_count_df1['Tags'][index])
kyph_rake_count_df1 = kyph_rake_count_df1[['Tags', 'Key Phrase Rake', 'Count']]
    
tag_new_yake = list(aa_yake['Tags'])
li_yake= []
for i in tag_new_yake:
    li_yake.append(get_singular(i))
    
aa_yake['Tags New'] = li_yake
aa_yake.drop(['Tags'], axis =1)
aa_yake = aa_yake[['Tags New', 'Count']]
bb_yake = aa_yake.groupby(aa_yake['Tags New']).aggregate({'Count':'sum'}).sort_values(by='Count').reset_index()
top_10_list_yake = list(bb_yake.nlargest(20, ['Count'])['Tags New'])

kyph_yake_count_df1['Tags'] = list(kyph_yake_count_df1['Key Phrase Yake'].str.findall('|'.join(top_10_list_yake)))
kyph_yake_count_df1 = kyph_yake_count_df1.dropna(subset=['Tags'])
for index, row in kyph_yake_count_df1.iterrows():
    kyph_yake_count_df1['Tags'][index] = ','.join(str(item) for item in kyph_yake_count_df1['Tags'][index])
kyph_yake_count_df1 = kyph_yake_count_df1[['Tags', 'Key Phrase Yake', 'Count']]
    
tag_new_multi_rake = list(aa_multi_rake['Tags'])
li_multi_rake= []
for i in tag_new_multi_rake:
    li_multi_rake.append(get_singular(i))
    
aa_multi_rake['Tags New'] = li_multi_rake
aa_multi_rake.drop(['Tags'], axis =1)
aa_multi_rake = aa_multi_rake[['Tags New', 'Count']]
bb_multi_rake = aa_multi_rake.groupby(aa_multi_rake['Tags New']).aggregate({'Count':'sum'}).sort_values(by='Count').reset_index()
top_10_list_multi_rake = list(bb_multi_rake.nlargest(20, ['Count'])['Tags New'])

kyph_multi_rake_count_df1['Tags'] = list(kyph_multi_rake_count_df1['Key Phrase Multi Rake'].str.findall('|'.join(top_10_list_multi_rake)))
kyph_multi_rake_count_df1 = kyph_multi_rake_count_df1.dropna(subset=['Tags'])
for index, row in kyph_multi_rake_count_df1.iterrows():
    kyph_multi_rake_count_df1['Tags'][index] = ','.join(str(item) for item in kyph_multi_rake_count_df1['Tags'][index])
kyph_multi_rake_count_df1 = kyph_multi_rake_count_df1[['Tags', 'Key Phrase Multi Rake', 'Count']]
    
tag_new_summa = list(aa_summa['Tags'])
li_summa= []
for i in tag_new_summa:
    li_summa.append(get_singular(i))
    
aa_summa['Tags New'] = li_summa
aa_summa.drop(['Tags'], axis =1)
aa_summa = aa_summa[['Tags New', 'Count']]
bb_summa = aa_summa.groupby(aa_summa['Tags New']).aggregate({'Count':'sum'}).sort_values(by='Count').reset_index()
top_10_list_summa = list(bb_summa.nlargest(20, ['Count'])['Tags New'])

kyph_summa_count_df1['Tags'] = list(kyph_summa_count_df1['Key Phrase Summa'].str.findall('|'.join(top_10_list_summa)))
kyph_summa_count_df1 = kyph_summa_count_df1.dropna(subset=['Tags'])
for index, row in kyph_summa_count_df1.iterrows():
    kyph_summa_count_df1['Tags'][index] = ','.join(str(item) for item in kyph_summa_count_df1['Tags'][index])
kyph_summa_count_df1 = kyph_summa_count_df1[['Tags', 'Key Phrase Summa', 'Count']]

path = r""

with pd.ExcelWriter(path, engine='openpyxl', mode='a') as writer:  
    df.to_excel(writer, sheet_name = 'Custom Keywords')
    kyph_rake_count_df.to_excel(writer, sheet_name = 'KP Rake Word Count')
    kyph_yake_count_df.to_excel(writer, sheet_name = 'KP Yake Word Count')
    kyph_multi_rake_count_df.to_excel(writer, sheet_name = 'KP Multi Rake Word Count')
    kyph_summa_count_df.to_excel(writer, sheet_name = 'KP Summa Word Count')
    kyph_count_df1.to_excel(writer, sheet_name = 'KP Tagging')
    kyph_rake_count_df1.to_excel(writer, sheet_name = 'KP Rake Tagging')
    kyph_yake_count_df1.to_excel(writer, sheet_name = 'KP Yake Tagging')
    kyph_multi_rake_count_df1.to_excel(writer, sheet_name = 'KP Multi Rake Tagging')
    kyph_summa_count_df1.to_excel(writer, sheet_name = 'KP Summa Tagging')
