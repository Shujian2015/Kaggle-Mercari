# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import warnings
warnings.filterwarnings("ignore")

import os ; os.environ['OMP_NUM_THREADS'] = '4'

import gc
import time;  start_time = time.time()
from time import gmtime, strftime
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import normalize
import sys

#Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
#until Kaggle admins fix the wordbatch pip package installation
#sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch

from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

from nltk.corpus import stopwords
import re

# ----------------------------------------------------------------------
## Settings:

# General
develop= True # develop = False
TEST_SIZE = 0.05
NUM_BRANDS = 4500
NUM_CATEGORIES = 1250
TEST_CHUNK_SIZE = 350000

# Tensorflow
TF_EPOCH = 4
TF_CHUNK_SIZE = 5000

# FM



def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


# ----------------------------------------------------------------------
## Helper function for FM and FTRL

def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

# Use word embeddings
wv ='../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt'
def loadWordVecs():
    embeddings_index = {}
    f = open(wv)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def sent2vec(s, embeddings_index): # this function creates a normalized vector for the whole sentence
    words = str(s).lower()
    '''
    # Good but too slow
    words = word_tokenize(words)
    words = [w for w in words if not w in stopwords.words('english')]
    words = [w for w in words if w.isalpha()]

    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    '''
    words = words.split()   
    M = []
    for w in words:
        M.append(embeddings_index.get(w, np.zeros(50)))
    
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(50)
    return v / np.sqrt((v ** 2).sum())


def fit_dummy(list_data):
    n_data = len(list_data)
    n_cat = 0
    d_data = {}
    for cat in list_data:
        if cat not in d_data:
            d_data[cat] = n_cat
            n_cat += 1
    d_data['<NOT_SHOWN>'] = n_cat
    res = np.zeros(shape=(n_data, n_cat + 1))
    for i, v in enumerate(list_data):
        res[i, d_data[v]] = 1
    return csr_matrix(res), d_data
    
def transform_dummy(list_data, d_data):
    n_data = len(list_data)
    n_cat = len(d_data)
    res = np.zeros(shape=(n_data, n_cat))
    for i, v in enumerate(list_data):
        if v in d_data:
            res[i, d_data[v]] = 1
        else:
            res[i, n_cat - 1] = 1
    return csr_matrix(res)
    
    
    
    
# ----------------------------------------------------------------------------------------------
## FM and FTRL -Wordbatch models    

#  https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44944   
def wordbatch_algo():
    import time
    
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    train = pd.read_table('../input/train.tsv', engine='c')

    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    '''
    nrow_test = train.shape[0]  # -dftt.shape[0]
    dftt = train[(train.price < 1.0)]
    train = train.drop(train[(train.price < 1.0)].index)
    del dftt['price']
    nrow_train = train.shape[0]
    # print(nrow_train, nrow_test)
    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, dftt, test])
    submission: pd.DataFrame = test[['test_id']]
    '''
    y = np.log1p(train["price"])
    
    nrow_train = train.shape[0]
    

    # Training
    train['general_cat'], train['subcat_1'], train['subcat_2'] = \
        zip(*train['category_name'].apply(lambda x: split_cat(x)))
    train.drop('category_name', axis=1, inplace=True)    
    print('[{}] Split categories completed.'.format(time.time() - start_time))
    
    handle_missing_inplace(train)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))
    
    cutting(train)
    print('[{}] Cut completed.'.format(time.time() - start_time))

    to_categorical(train)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))    

    # Add some new features:
    X_len_desc = train['item_description'].apply(lambda x: len(x)).as_matrix().reshape(-1, 1)
    X_len_name = train['name'].apply(lambda x: len(x)).as_matrix().reshape(-1, 1)

    print('[{}] Length of text completed.'.format(time.time() - start_time))     
    
    # Name
    wb_name = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                  "idf": None,
                                                                  }), procs=8)    
    
    wb_name.dictionary_freeze= True
    wb_name.fit(train['name'])
    X_name = wb_name.transform(train['name'])
    
    # X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))    
    
    wb_cat1 = CountVectorizer()
    wb_cat2 = CountVectorizer()
    wb_cat3 = CountVectorizer()
    wb_cat1.fit(train['general_cat'])
    wb_cat2.fit(train['subcat_1'])
    wb_cat3.fit(train['subcat_2'])
    
    X_category1 = wb_cat1.transform(train['general_cat'])
    X_category2 = wb_cat2.transform(train['subcat_1'])
    X_category3 = wb_cat3.transform(train['subcat_2'])       
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))    
    
    # wb= wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 3, "hash_ngrams_weights": [1.0, 1.0, 0.5],
    wb_desc = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                                  "idf": None})
                             , procs=8)
    wb_desc.dictionary_freeze= True
    wb_desc.fit(train['item_description'])
    X_description = wb_desc.transform(train['item_description'])

    # X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))    
    
    lb = LabelBinarizer(sparse_output=True)
    lb.fit(train['brand_name'])
    X_brand = lb.transform(train['brand_name'])
    print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))
    

    X_cond, d_cond = fit_dummy(train['item_condition_id'].tolist())
    X_ship, d_ship = fit_dummy(train['shipping'].tolist())
    
    print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))

    del train
    gc.collect()
    
    
    print(X_cond.shape, X_ship.shape, X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
          X_name.shape) 
    sparse_merge = hstack((X_cond, X_ship, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()
    
    print('[{}] Create sparse merge completed'.format(time.time() - start_time))
    del X_description, X_brand, X_category1, X_category2, X_category3, X_name
    gc.collect()
    
    # Remove features with document frequency <=1

    print(sparse_merge.shape)
    mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    sparse_merge = sparse_merge[:, mask]
    print(sparse_merge.shape)
    X = sparse_merge
    
    
    
    # ---------------------------------------
    # FM model
    train_X, train_y = X, y
    if develop:
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=TEST_SIZE, random_state=100)


    model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=train_X.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=17, inv_link="identity", threads=4)

    model.fit(train_X, train_y)
    gc.collect()
    print('[{}] Train FM_FTRL completed'.format(time.time() - start_time))
    print('-' * 20)
    if develop:
        preds = model.predict(X=valid_X)
        print("->>>>  FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))
    
    # ---------------------------------------
    # FTRL model
    model2 = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=train_X.shape[1], iters=20, inv_link="identity", threads=1)
    # del X; gc.collect()
    model2.fit(train_X, train_y)
    print('[{}] Train FTRL completed'.format(time.time() - start_time))
    if develop:
        preds = model2.predict(X=valid_X)
        print("->>>>  FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))
        
        
    # Clear variables:
    del X, train_X, sparse_merge
    gc.collect()          
    # ---------------------------------------
    # Testing
    print(' FM/FTRL: ...reading the test data...')
    predsFM = []
    predsF = []
    # https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/48378#279887
    def load_test():
        for df in pd.read_csv('../input/test.tsv', sep='\t', chunksize=TEST_CHUNK_SIZE):
            yield df    

    for test in load_test():    
        test['general_cat'], test['subcat_1'], test['subcat_2'] = \
            zip(*test['category_name'].apply(lambda x: split_cat(x)))
        test.drop('category_name', axis=1, inplace=True)        

        handle_missing_inplace(test)
        #print('[{}] Handle missing completed.'.format(time.time() - start_time))

        cutting(test)
        # print('[{}] Cut completed.'.format(time.time() - start_time))

        to_categorical(test)
        # print('[{}] Convert categorical completed'.format(time.time() - start_time))        

        # Add some new features:
        X_len_desc_test = test['item_description'].apply(lambda x: len(x)).as_matrix().reshape(-1, 1)
        X_len_name_test = test['name'].apply(lambda x: len(x)).as_matrix().reshape(-1, 1)    

        X_name_test = wb_name.transform(test['name'])
        # X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

        X_category1_test = wb_cat1.transform(test['general_cat'])
        X_category2_test = wb_cat2.transform(test['subcat_1'])
        X_category3_test = wb_cat3.transform(test['subcat_2'])   


        X_description_test = wb_desc.transform(test['item_description'])
        # X_description_test = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

        X_brand_test = lb.transform(test['brand_name'])

        X_cond_test = transform_dummy(test['item_condition_id'].tolist(), d_cond)  
        X_ship_test = transform_dummy(test['shipping'].tolist(), d_ship)


        #del test
        #gc.collect()

        X_test = hstack((X_cond_test, X_ship_test, X_description_test, X_brand_test, X_category1_test, \
                         X_category2_test, X_category3_test, X_name_test)).tocsr()
        X_test = X_test[:, mask]
        
        
        predsFM_batch = model.predict(X_test)
        predsFM += np.array(predsFM_batch).flatten().tolist() 
        
        predsF_batch = model2.predict(X_test)
        predsF += np.array(predsF_batch).flatten().tolist() 
        gc.collect()
        
    print(np.array(predsFM))
    print('-' * 20)    

    print(np.array(predsF))
    print('-' * 20)
    
    # del train_X, train_y
    # del X_test    
    
    return np.array(predsFM), np.array(predsF)
    
    
    

# ----------------------------------------------------------------------------------------------
## Helper functions for neural networks

# Refer to: https://www.kaggle.com/lscoelho/tensorflow-starter-conv1d-emb-0-43839-lb-v08?scriptVersionId=2084098
import re
from time import time
from collections import Counter

import tensorflow as tf
import pandas as pd
import numpy as np

from nltk.stem.porter import PorterStemmer
from fastcache import clru_cache as lru_cache

# https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/48378#274654
import pyximport
pyximport.install()
session_conf = tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=1)

np.random.seed(125)


t_start = time()

stemmer = PorterStemmer()

@lru_cache(1024)
def stem(s):
    return stemmer.stem(s)

whitespace = re.compile(r'\s+')
non_letter = re.compile(r'\W+')

def tokenize(text):
    text = text.lower()
    text = non_letter.sub(' ', text)

    tokens = []

    for t in text.split():
        t = stem(t)
        tokens.append(t)

    return tokens

class Tokenizer:
    def __init__(self, min_df=10, tokenizer=str.split):
        self.min_df = min_df
        self.tokenizer = tokenizer
        self.doc_freq = None
        self.vocab = None
        self.vocab_idx = None
        self.max_len = None

    def fit_transform(self, texts):
        tokenized = []
        doc_freq = Counter()
        n = len(texts)

        for text in texts:
            sentence = self.tokenizer(text)
            tokenized.append(sentence)
            doc_freq.update(set(sentence))

        vocab = sorted([t for (t, c) in doc_freq.items() if c >= self.min_df])
        vocab_idx = {t: (i + 1) for (i, t) in enumerate(vocab)}
        doc_freq = [doc_freq[t] for t in vocab]

        self.doc_freq = doc_freq
        self.vocab = vocab
        self.vocab_idx = vocab_idx

        max_len = 0
        result_list = []
        for text in tokenized:
            text = self.text_to_idx(text)
            max_len = max(max_len, len(text))
            result_list.append(text)

        self.max_len = max_len
        result = np.zeros(shape=(n, max_len), dtype=np.int32)
        for i in range(n):
            text = result_list[i]
            result[i, :len(text)] = text

        return result    

    def text_to_idx(self, tokenized):
        return [self.vocab_idx[t] for t in tokenized if t in self.vocab_idx]

    def transform(self, texts):
        n = len(texts)
        result = np.zeros(shape=(n, self.max_len), dtype=np.int32)

        for i in range(n):
            text = self.tokenizer(texts[i])
            text = self.text_to_idx(text)[:self.max_len]
            result[i, :len(text)] = text

        return result
    
    def vocabulary_size(self):
        return len(self.vocab) + 1    
        

# ----------------------------------------------------------------------------------------------
## CNN method in Tensorflow

def tf_method(): 
    print(' TF: ...reading train data...')
    # df_train = pd.read_csv('../input/train.tsv', sep='\t', nrows = 1000) ## TODO
    df_train = pd.read_csv('../input/train.tsv', sep='\t')
    df_train = df_train[df_train.price != 0].reset_index(drop=True)


    df_train.name.fillna('unkname', inplace=True)
    df_train.category_name.fillna('unk_cat', inplace=True)
    df_train.brand_name.fillna('unk_brand', inplace=True)
    df_train.item_description.fillna('nodesc', inplace=True)

    
    print(' TF: ...processing my own features...')
    
    # Average price of each label
    my_features = [('category_name', 'cat_mean'),
                   ('item_condition_id', 'cond_mean'),
                   ('brand_name', 'brand_mean'),
                   ('shipping', 'ship_mean'),]
    my_feat_df = []
    
    for (feat_name, mean_name) in my_features:
        tmp_feat_df = df_train['price'].groupby(df_train[feat_name]).mean()
        tmp_feat_df = pd.DataFrame({feat_name:tmp_feat_df.index, mean_name:tmp_feat_df.values})
        df_train = df_train.merge(tmp_feat_df, on=[feat_name], how='left')
        my_feat_df.append(tmp_feat_df) 

    
    df_train['len_desc'] = df_train['item_description'].apply(lambda x: len(x))
    df_train['len_name'] = df_train['name'].apply(lambda x: len(x))

    
    # Normalize data
    normalizers = []
    for col in ['cat_mean', 'cond_mean', 'brand_mean', 'ship_mean', 'len_desc', 'len_name']:
        normalizer = MinMaxScaler(feature_range=(-1, 1)).fit(df_train[[col]].values)
        df_train[col] = normalizer.transform(np.nan_to_num(df_train[[col]].values))
        normalizers.append(normalizer)
        
    X_cat_mean = df_train.cat_mean.astype('float32').values.reshape(-1, 1)        
    X_cond_mean = df_train.cond_mean.astype('float32').values.reshape(-1, 1)            
    X_brand_mean = df_train.brand_mean.astype('float32').values.reshape(-1, 1)    
    X_ship_mean = df_train.ship_mean.astype('float32').values.reshape(-1, 1)             
    X_len_desc = df_train.len_desc.astype('float32').values.reshape(-1, 1)         
    X_len_name = df_train.len_name.astype('float32').values.reshape(-1, 1)     
    
    
    # Dealing with price
    price = df_train.pop('price')
    y = np.log1p(price.values)
    mean = y.mean()
    std = y.std()
    y = (y - mean) / std
    y = y.reshape(-1, 1)
    
    
    
    print(' TF: ...processing category...')

    def paths(tokens):
        all_paths = ['/'.join(tokens[0:(i+1)]) for i in range(len(tokens))]
        return ' '.join(all_paths)

    @lru_cache(1024)
    def cat_process(cat):
        cat = cat.lower()
        cat = whitespace.sub('', cat)
        split = cat.split('/')
        return paths(split)

    df_train.category_name = df_train.category_name.apply(cat_process)

    cat_tok = Tokenizer(min_df=55)
    X_cat = cat_tok.fit_transform(df_train.category_name)
    cat_voc_size = cat_tok.vocabulary_size()


    print(' TF: ...processing title...')

    name_tok = Tokenizer(min_df=10, tokenizer=tokenize)
    X_name = name_tok.fit_transform(df_train.name)
    name_voc_size = name_tok.vocabulary_size()


    print(' TF: ...processing description...')

    desc_num_col = 53 #v0 40
    desc_tok = Tokenizer(min_df=50, tokenizer=tokenize)
    X_desc = desc_tok.fit_transform(df_train.item_description)
    X_desc = X_desc[:, :desc_num_col]
    desc_voc_size = desc_tok.vocabulary_size()


    print(' TF: ...processing brand...')

    df_train.brand_name = df_train.brand_name.str.lower()
    df_train.brand_name = df_train.brand_name.str.replace(' ', '_')

    brand_cnt = Counter(df_train.brand_name[df_train.brand_name != 'unk_brand'])
    brands = sorted(b for (b, c) in brand_cnt.items() if c >= 50)
    brands_idx = {b: (i + 1) for (i, b) in enumerate(brands)}

    X_brand = df_train.brand_name.apply(lambda b: brands_idx.get(b, 0))
    X_brand = X_brand.values.reshape(-1, 1) 
    brand_voc_size = len(brands) + 1


    print(' TF: ...processing other features...')

    X_item_cond = (df_train.item_condition_id - 1).astype('uint8').values.reshape(-1, 1)
    X_shipping = df_train.shipping.astype('float32').values.reshape(-1, 1)

    print(' TF: ...defining the model...')

    def prepare_batches(seq, step):
        n = len(seq)
        res = []
        for i in range(0, n, step):
            res.append(seq[i:i+step])
        return res

    def conv1d(inputs, num_filters, filter_size, padding='same'):
        he_std = np.sqrt(2 / (filter_size * num_filters))
        out = tf.layers.conv1d(
            inputs=inputs, filters=num_filters, padding=padding,
            kernel_size=filter_size,
            activation=tf.nn.relu, 
            kernel_initializer=tf.random_normal_initializer(stddev=he_std))
        return out

    def dense(X, size, reg=0.0, activation=None):
        he_std = np.sqrt(2 / int(X.shape[1]))
        out = tf.layers.dense(X, units=size, activation=activation, 
                         kernel_initializer=tf.random_normal_initializer(stddev=he_std),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))
        return out

    def embed(inputs, size, dim):
        std = np.sqrt(2 / dim)
        emb = tf.Variable(tf.random_uniform([size, dim], -std, std))
        lookup = tf.nn.embedding_lookup(emb, inputs)
        return lookup




    name_embeddings_dim = 32
    name_seq_len = X_name.shape[1]
    desc_embeddings_dim = 32
    desc_seq_len = X_desc.shape[1]

    brand_embeddings_dim = 4

    cat_embeddings_dim = 15
    cat_seq_len = X_cat.shape[1]


    graph = tf.Graph()
    graph.seed = 1

    with graph.as_default():
        place_name = tf.placeholder(tf.int32, shape=(None, name_seq_len))
        place_desc = tf.placeholder(tf.int32, shape=(None, desc_seq_len))
        place_brand = tf.placeholder(tf.int32, shape=(None, 1))
        place_cat = tf.placeholder(tf.int32, shape=(None, cat_seq_len))
        place_ship = tf.placeholder(tf.float32, shape=(None, 1))
        place_cond = tf.placeholder(tf.uint8, shape=(None, 1))
        #---
        place_cat_mean = tf.placeholder(tf.float32, shape=(None, 1))    
        place_cond_mean = tf.placeholder(tf.float32, shape=(None, 1))            
        place_brand_mean = tf.placeholder(tf.float32, shape=(None, 1)) 
        place_ship_mean = tf.placeholder(tf.float32, shape=(None, 1))           
        place_len_desc = tf.placeholder(tf.float32, shape=(None, 1))         
        place_len_name = tf.placeholder(tf.float32, shape=(None, 1))         
        #---
        
        place_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        place_lr = tf.placeholder(tf.float32, shape=(), )

        name = embed(place_name, name_voc_size, name_embeddings_dim)
        desc = embed(place_desc, desc_voc_size, desc_embeddings_dim)
        brand = embed(place_brand, brand_voc_size, brand_embeddings_dim)
        cat = embed(place_cat, cat_voc_size, cat_embeddings_dim)

        name = conv1d(name, num_filters=13, filter_size=3)
        name = tf.layers.dropout(name, rate=0.5)
        name = tf.contrib.layers.flatten(name)
        print(' TF: ...', name.shape)

        desc = conv1d(desc, num_filters=12, filter_size=3)
        desc = tf.layers.dropout(desc, rate=0.5)
        desc = tf.contrib.layers.flatten(desc)
        print(' TF: ...', desc.shape)

        brand = tf.contrib.layers.flatten(brand)
        print(' TF: ...', brand.shape)

        cat = tf.layers.average_pooling1d(cat, pool_size=cat_seq_len, strides=1, padding='valid')
        cat = tf.contrib.layers.flatten(cat)
        print(' TF: ...', cat.shape)

        ship = place_ship
        print(' TF: ...', ship.shape)

        cond = tf.one_hot(place_cond, 5)
        cond = tf.contrib.layers.flatten(cond)
        print(' TF: ...', cond.shape)

        out = tf.concat([name, desc, brand, cat, ship, cond], axis=1)
        print(' TF: ...concatenated dim:', out.shape)

        # https://www.kaggle.com/valkling/mercari-rnn-2ridge-models-with-notes-0-42755
        out = dense(out, 512, activation=tf.nn.relu)
        out = tf.layers.dropout(out, rate=0.1)
        out = dense(out, 256, activation=tf.nn.relu)
        out = tf.layers.dropout(out, rate=0.1)
        out = dense(out, 128, activation=tf.nn.relu)
        out = tf.layers.dropout(out, rate=0.1)        
        out = dense(out, 64, activation=tf.nn.relu)
        out = tf.layers.dropout(out, rate=0.1)
        
        out = tf.concat([out, place_cat_mean, place_cond_mean, place_brand_mean, place_ship_mean, place_len_desc, place_len_name], axis=1)
        out = tf.layers.dropout(out, rate=0.5)
        out = dense(out, 1)

        loss = tf.losses.mean_squared_error(place_y, out)
        rmse = tf.sqrt(loss)
        opt = tf.train.AdamOptimizer(learning_rate=place_lr)
        train_step = opt.minimize(loss)

        init = tf.global_variables_initializer()

    session = tf.Session(config=None, graph=graph)
    session.run(init)


    print(' TF: ...training the model...')
    train_idx_shuffle = np.arange(X_name.shape[0])
    if develop:
        train_idx_shuffle, val_idx_shuffle = train_test_split(train_idx_shuffle, test_size=TEST_SIZE, random_state=100)

    for i in range(TF_EPOCH):
        t0 = time()
        np.random.seed(i)
        # train_idx_shuffle = np.arange(X_name.shape[0])
        np.random.shuffle(train_idx_shuffle)
        batches = prepare_batches(train_idx_shuffle, 500)
        # batches = np.array_split(train_idx_shuffle, 500)
        
        if i <= 2:
            lr = 0.0012
        else:
            lr = 0.0001

        for idx in batches:
            feed_dict = {
                place_name: X_name[idx],
                place_desc: X_desc[idx],
                place_brand: X_brand[idx],
                place_cat: X_cat[idx],
                place_cond: X_item_cond[idx],
                place_ship: X_shipping[idx],
                place_y: y[idx],
                place_lr: lr,
                place_cat_mean: X_cat_mean[idx],  
                place_cond_mean: X_cond_mean[idx],            
                place_brand_mean: X_brand_mean[idx], 
                place_ship_mean: X_ship_mean[idx],      
                place_len_desc: X_len_desc[idx], 
                place_len_name: X_len_name[idx],                   
            }
            session.run(train_step, feed_dict=feed_dict)

        took = time() - t0
        print(' TF: ......epoch %d took %.3fs' % (i, took))



    if develop: 
        # val_batches = np.array_split(val_idx_shuffle, 5000)
        val_batches = prepare_batches(val_idx_shuffle, 5000)
        val_preds = []
        
        for idx in val_batches:
            feed_dict = {
                place_name: X_name[idx],
                place_desc: X_desc[idx],
                place_brand: X_brand[idx],
                place_cat: X_cat[idx],
                place_cond: X_item_cond[idx],
                place_ship: X_shipping[idx],
                place_cat_mean: X_cat_mean[idx],  
                place_cond_mean: X_cond_mean[idx],            
                place_brand_mean: X_brand_mean[idx], 
                place_ship_mean: X_ship_mean[idx],      
                place_len_desc: X_len_desc[idx], 
                place_len_name: X_len_name[idx],                   
            }
            batch_pred = session.run(out, feed_dict=feed_dict)
            val_preds += [i[0] for i in batch_pred]
        val_preds = np.array(val_preds) * std + mean
        print(" TF: ----->>>>  TF dev RMSLE:", rmsle(np.expm1(y[val_idx_shuffle, :].flatten() * std + mean), np.expm1(val_preds)))        
        

    print(' TF: ...reading the test data...')
    y_pred = []
    # https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/48378#279887
    def load_test():
        for df in pd.read_csv('../input/test.tsv', sep='\t', chunksize=TEST_CHUNK_SIZE):
            yield df    

    for df_test in load_test():
        gc.collect()
        print(' TF: ......applying the model to a chunk of test data...')
        df_test.name.fillna('unkname', inplace=True)
        df_test.category_name.fillna('unk_cat', inplace=True)
        df_test.brand_name.fillna('unk_brand', inplace=True)
        df_test.item_description.fillna('nodesc', inplace=True)


        # My own features: -----
        for i, (feat_name, _) in enumerate(my_features):
            tmp_feat_df = my_feat_df[i]
            df_test = df_test.merge(tmp_feat_df, on=[feat_name], how='left')

        df_test['len_desc'] = df_test['item_description'].apply(lambda x: len(x))
        df_test['len_name'] = df_test['name'].apply(lambda x: len(x))

        for i, col in enumerate(['cat_mean', 'cond_mean', 'brand_mean', 'ship_mean', 'len_desc', 'len_name']):
            normalizer = normalizers[i]
            df_test[col] = normalizer.transform(np.nan_to_num(df_test[[col]].values))
            normalizers.append(normalizer)    

        X_cat_mean_test = df_test.cat_mean.astype('float32').values.reshape(-1, 1)        
        X_cond_mean_test = df_test.cond_mean.astype('float32').values.reshape(-1, 1)            
        X_brand_mean_test = df_test.brand_mean.astype('float32').values.reshape(-1, 1)    
        X_ship_mean_test = df_test.ship_mean.astype('float32').values.reshape(-1, 1)             
        X_len_desc_test = df_test.len_desc.astype('float32').values.reshape(-1, 1)         
        X_len_name_test = df_test.len_name.astype('float32').values.reshape(-1, 1)           
        # --------

        df_test.category_name = df_test.category_name.apply(cat_process)
        df_test.brand_name = df_test.brand_name.str.lower()
        df_test.brand_name = df_test.brand_name.str.replace(' ', '_')

        X_cat_test = cat_tok.transform(df_test.category_name)
        X_name_test = name_tok.transform(df_test.name)

        X_desc_test = desc_tok.transform(df_test.item_description)
        X_desc_test = X_desc_test[:, :desc_num_col]

        X_item_cond_test = (df_test.item_condition_id - 1).astype('uint8').values.reshape(-1, 1)
        X_shipping_test = df_test.shipping.astype('float32').values.reshape(-1, 1)

        X_brand_test = df_test.brand_name.apply(lambda b: brands_idx.get(b, 0))
        X_brand_test = X_brand_test.values.reshape(-1, 1)


        n_test = len(df_test)
        test_idx = np.arange(n_test)
        batches = prepare_batches(test_idx, TF_CHUNK_SIZE)
        # batches = np.array_split(test_idx, TF_CHUNK_SIZE)
        y_preds_batch = []
        
        for idx in batches:
            feed_dict = {
                place_name: X_name_test[idx],
                place_desc: X_desc_test[idx],
                place_brand: X_brand_test[idx],
                place_cat: X_cat_test[idx],
                place_cond: X_item_cond_test[idx],
                place_ship: X_shipping_test[idx],
                place_cat_mean: X_cat_mean_test[idx],  
                place_cond_mean: X_cond_mean_test[idx],            
                place_brand_mean: X_brand_mean_test[idx], 
                place_ship_mean: X_ship_mean_test[idx],      
                place_len_desc: X_len_desc_test[idx], 
                place_len_name: X_len_name_test[idx],            
            }
            
            
            batch_pred = session.run(out, feed_dict=feed_dict)
            y_preds_batch += [i[0] for i in batch_pred]
        y_pred += y_preds_batch    
    y_pred = np.array(y_pred) * std + mean

    print(' TF: ...writing the results...')

    print(y_pred)
    print('-' * 20)
    return y_pred  ## TODO
    
    
# ----------------------------------------------------------------------------------------------
## Main function. Call all the methods and weighted average them. Save to .csv

def main():
    submission = pd.read_table('../input/test.tsv', engine='c', usecols=['test_id'])
    
    predsTF = tf_method(); gc.collect()
    predsFM, predsF = wordbatch_algo()
    
    preds = (predsF * 0.10 + predsTF * 0.30 + predsFM * 0.60)
    
    submission['price'] = np.expm1(preds)
    submission.to_csv("submission_wordbatch_ftrl_fm_tf.csv", index=False)
    
 
if __name__ == '__main__':
    main()