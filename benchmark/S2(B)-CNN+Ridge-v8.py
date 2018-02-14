# NN models
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


# ----------------------------------------------------------------------
## Settings:

# General
develop= True
TEST_SIZE = 0.05
SPLIT_SEED = 100
NUM_BRANDS = 4500
NUM_CATEGORIES = 1250
TEST_CHUNK_SIZE = 350000

# Tensorflow
TF_EPOCH = 4

TF_CHUNK_SIZE = 5000
BATCH_SIZE = 500
DROPOUT = 0.5
'''
LR_CUT = 2 # Change lr after this epoch
LR_INIT = 0.0012
LR_END = 0.0001
'''
LRs = [0.01] * 3 + [0.001]

name_embeddings_dim = 32
desc_embeddings_dim = 32
brand_embeddings_dim = 4
cat_embeddings_dim = 15   

# FM and FTRL
FM_iter = 17
FTRL_iter = 50

# ----------------------------------------------------------------------------------------------
# Load test by chunk
# https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/48378#279887
def load_test():
    for df in pd.read_csv('../input/test.tsv', sep='\t', chunksize=TEST_CHUNK_SIZE):
        yield df  
        


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))
    
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# NN model
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
## Helper functions for neural networks

# Refer to: https://www.kaggle.com/lscoelho/tensorflow-starter-conv1d-emb-0-43839-lb-v08?scriptVersionId=2084098
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
        # t = stem(t)  # TODO
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
        

# --------------------
# https://www.kaggle.com/golubev/naive-xgboost-v2?scriptVersionId=1793318

def count_words(key):
    return len(str(key).split())

def count_numbers(key):
    return sum(c.isalpha() for c in key)

def count_upper(key):
    return sum(c.isupper() for c in key)

def get_mean(df, name, target, alpha=0):
    group = df.groupby(name)[target].agg([np.sum, np.size])
    mean = train[target].mean()
    series = (group['sum'] + mean*alpha)/(group['size']+alpha)
    series.name = name + '_mean'
    return series.to_frame().reset_index()

def add_words(df, name, length):
    x_data = []
    for x in df[name].values:
        x_row = np.ones(length, dtype=np.uint16)*0
        for xi, i in zip(list(str(x)), np.arange(length)):
            x_row[i] = ord(xi)
        x_data.append(x_row)
    return pd.concat([df, pd.DataFrame(x_data, columns=[name+str(c) for c in range(length)]).astype(np.uint16)], axis=1)


# ----------------------------------------------------------------------------------------------
## CNN method in Tensorflow

def tf_method(df_test): 
    print(' TF: ...reading train data...')
    # df_train = pd.read_csv('../input/train.tsv', sep='\t', nrows = 1000) ## TODO
    df_train = pd.read_csv('../input/train.tsv', sep='\t')
    df_train = df_train[df_train.price != 0].reset_index(drop=True)


    df_train.name.fillna('unkname', inplace=True)
    df_train.category_name.fillna('unk_cat', inplace=True)
    df_train.brand_name.fillna('unk_brand', inplace=True)
    df_train.item_description.fillna('nodesc', inplace=True)



     #----------------    
    print(' TF: ...processing my own features...')
    
    df_train['log_price'] = np.log1p(df_train['price'])
    
    # Average price of each label
    my_features = [('category_name', 'cat_mean'),
                   ('item_condition_id', 'cond_mean'),
                   ('brand_name', 'brand_mean'),
                   ('shipping', 'ship_mean'),]
    my_feat_df = []
    
    for (feat_name, mean_name) in my_features:
        tmp_feat_df = df_train['log_price'].groupby(df_train[feat_name]).mean()
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
    
    '''
    #----------------        
    # New features from: https://www.kaggle.com/golubev/naive-xgboost-v2?scriptVersionId=1793318
    for c in ['item_description', 'name']:
        df_train[c + '_c_words'] = df_train[c].apply(count_words)
        df_train[c + '_c_upper'] = df_train[c].apply(count_upper)
        df_train[c + '_c_numbers'] = df_train[c].apply(count_numbers)
        df_train[c + '_len'] = df_train[c].str.len()
        df_train[c + '_mean_len_words'] = df_train[c + '_len'] / df_train[c + '_c_words']
        df_train[c + '_mean_upper'] = df_train[c + '_len'] / df_train[c + '_c_upper']
        df_train[c + '_mean_numbers'] = df_train[c + '_len'] / df_train[c + '_c_numbers']    
    
    
    # Normalize data
    normalizers2 = []
    for c in ['item_description', 'name']:
        for col in ['_mean_len_words', '_mean_upper', '_mean_numbers']:
            normalizer = MinMaxScaler(feature_range=(-1, 1)).fit(np.nan_to_num(df_train[[c + col]].values))
            df_train[c + col] = normalizer.transform(np.nan_to_num(df_train[[c + col]].values))
            normalizers2.append(normalizer)
  
  
    X_name_mean_len_words = df_train.name_mean_len_words.astype('float32').values.reshape(-1, 1) 
    X_name_mean_upper = df_train.name_mean_upper.astype('float32').values.reshape(-1, 1) 
    X_name_mean_numbers = df_train.name_mean_numbers.astype('float32').values.reshape(-1, 1) 
    X_desc_mean_len_words = df_train.item_description_mean_len_words.astype('float32').values.reshape(-1, 1) 
    X_desc_mean_upper = df_train.item_description_mean_upper.astype('float32').values.reshape(-1, 1) 
    X_desc_mean_numbers = df_train.item_description_mean_numbers.astype('float32').values.reshape(-1, 1) 
    # print(df_train[['name_mean_len_words']])
    # ----------------
    '''
    
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

    
    name_seq_len = X_name.shape[1]
    desc_seq_len = X_desc.shape[1]
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
        place_len_name  = tf.placeholder(tf.float32, shape=(None, 1))
        
        my_feat = [place_cat_mean, place_cond_mean, place_brand_mean, place_ship_mean, place_len_desc, place_len_name]
        #---
        place_desc_mean_len_words = tf.placeholder(tf.float32, shape=(None, 1))
        place_desc_mean_upper = tf.placeholder(tf.float32, shape=(None, 1))
        place_desc_mean_numbers = tf.placeholder(tf.float32, shape=(None, 1))
        place_name_mean_len_words = tf.placeholder(tf.float32, shape=(None, 1))
        place_name_mean_upper = tf.placeholder(tf.float32, shape=(None, 1))
        place_name_mean_numbers = tf.placeholder(tf.float32, shape=(None, 1))

        nx_feat = [place_desc_mean_len_words, place_desc_mean_upper, place_desc_mean_numbers, place_name_mean_len_words, place_name_mean_upper, place_name_mean_numbers]
        # ----
        
        place_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        place_lr = tf.placeholder(tf.float32, shape=(), )

        name = embed(place_name, name_voc_size, name_embeddings_dim)
        desc = embed(place_desc, desc_voc_size, desc_embeddings_dim)
        brand = embed(place_brand, brand_voc_size, brand_embeddings_dim)
        cat = embed(place_cat, cat_voc_size, cat_embeddings_dim)

        # CNN for name and description:
        name = conv1d(name, num_filters=name_seq_len, filter_size=3)
        name = tf.layers.dropout(name, rate=DROPOUT)
        name = tf.layers.average_pooling1d(name, pool_size=name_seq_len, strides=1, padding='valid')
        name = tf.contrib.layers.flatten(name)
        print(' TF: ...', name.shape)

        desc = conv1d(desc, num_filters=desc_seq_len, filter_size=3)
        desc = tf.layers.dropout(desc, rate=DROPOUT)
        desc = tf.layers.average_pooling1d(desc, pool_size=desc_seq_len, strides=1, padding='valid')
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
        #out = tf.layers.dropout(out, rate=0.1)        
        out = dense(out, 256, activation=tf.nn.relu)
        #out = tf.layers.dropout(out, rate=0.1)
        out = dense(out, 128, activation=tf.nn.relu)
        # out = tf.layers.dropout(out, rate=0.1)        
        out = dense(out, 64, activation=tf.nn.relu)
        #out = tf.layers.dropout(out, rate=0.1)
        
        out = tf.concat([out] + my_feat, axis=1)
        # out = tf.concat([out] + my_feat + nx_feat, axis=1) # Add extra features to out
        # out = tf.layers.dropout(out, rate=0.5)
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
        train_idx_shuffle, val_idx_shuffle = train_test_split(train_idx_shuffle, test_size=TEST_SIZE, random_state=SPLIT_SEED)

    for i in range(TF_EPOCH):
        t0 = time()
        np.random.seed(i)
        np.random.shuffle(train_idx_shuffle)
        batches = prepare_batches(train_idx_shuffle, BATCH_SIZE)
        
        lr = LRs[i]
        '''
        if i <= LR_CUT:  
            lr = LR_INIT 
        else:
            lr = LR_END
        '''
        
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
        val_batches = prepare_batches(val_idx_shuffle, TF_CHUNK_SIZE)
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
    

    if True:
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
 
        

        X_cat_mean_test = df_test.cat_mean.astype('float32').values.reshape(-1, 1)        
        X_cond_mean_test = df_test.cond_mean.astype('float32').values.reshape(-1, 1)            
        X_brand_mean_test = df_test.brand_mean.astype('float32').values.reshape(-1, 1)    
        X_ship_mean_test = df_test.ship_mean.astype('float32').values.reshape(-1, 1)             
        X_len_desc_test = df_test.len_desc.astype('float32').values.reshape(-1, 1)         
        X_len_name_test = df_test.len_name.astype('float32').values.reshape(-1, 1)           

        '''
        #----------------        
        # New features from: https://www.kaggle.com/golubev/naive-xgboost-v2?scriptVersionId=1793318
        for c in ['item_description', 'name']:
            df_test[c + '_c_words'] = df_test[c].apply(count_words)
            df_test[c + '_c_upper'] = df_test[c].apply(count_upper)
            df_test[c + '_c_numbers'] = df_test[c].apply(count_numbers)
            df_test[c + '_len'] = df_test[c].str.len()
            df_test[c + '_mean_len_words'] = df_test[c + '_len'] / df_test[c + '_c_words']
            df_test[c + '_mean_upper'] = df_test[c + '_len'] / df_test[c + '_c_upper']
            df_test[c + '_mean_numbers'] = df_test[c + '_len'] / df_test[c + '_c_numbers']    
        
        
        # Normalize data
        for i, col in enumerate(['item_description_mean_len_words', 'item_description_mean_upper', 'item_description_mean_numbers', 'name_mean_len_words', 'name_mean_upper', 'name_mean_numbers']):
            normalizer = normalizers2[i]
            df_test[col] = normalizer.transform(np.nan_to_num(df_test[[col]].values))

      
      
        X_name_mean_len_words_test = df_test.name_mean_len_words.astype('float32').values.reshape(-1, 1) 
        X_name_mean_upper_test =     df_test.name_mean_upper.astype('float32').values.reshape(-1, 1) 
        X_name_mean_numbers_test =   df_test.name_mean_numbers.astype('float32').values.reshape(-1, 1) 
        X_desc_mean_len_words_test = df_test.item_description_mean_len_words.astype('float32').values.reshape(-1, 1) 
        X_desc_mean_upper_test =     df_test.item_description_mean_upper.astype('float32').values.reshape(-1, 1) 
        X_desc_mean_numbers_test =   df_test.item_description_mean_numbers.astype('float32').values.reshape(-1, 1)         
        '''

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
    return y_pred 
    

    
    
    
    
    

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# Ridge model
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

import multiprocessing as mp
import pandas as pd
from time import time
from scipy.sparse import csr_matrix
import os
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gc
from sklearn.base import BaseEstimator, TransformerMixin
import re
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'forkserver'

INPUT_PATH = r'../input'


def dameraulevenshtein(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.

    This method has not been modified from the original.
    Source: http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/

    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.

    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.

    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.

    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2

    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


class SymSpell:
    def __init__(self, max_edit_distance=3, verbose=0):
        self.max_edit_distance = max_edit_distance
        self.verbose = verbose
        # 0: top suggestion
        # 1: all suggestions of smallest edit distance
        # 2: all suggestions <= max_edit_distance (slower, no early termination)

        self.dictionary = {}
        self.longest_word_length = 0

    def get_deletes_list(self, w):
        """given a word, derive strings with up to max_edit_distance characters
           deleted"""

        deletes = []
        queue = [w]
        for d in range(self.max_edit_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c + 1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue

        return deletes

    def create_dictionary_entry(self, w):
        '''add word and its derived deletions to dictionary'''
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections,
        # frequency of word in corpus)
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0], self.dictionary[w][1] + 1)
        else:
            self.dictionary[w] = ([], 1)
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1] == 1:
            # first appearance of word in corpus
            # n.b. word may already be in dictionary as a derived word
            # (deleting character from a real word)
            # but counter of frequency of word in corpus is not incremented
            # in those cases)
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list
                    self.dictionary[item][0].append(w)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = ([w], 0)

        return new_real_word_added

    def create_dictionary_from_arr(self, arr, token_pattern=r'[a-z]+'):
        total_word_count = 0
        unique_word_count = 0

        for line in arr:
            # separate by words by non-alphabetical characters
            words = re.findall(token_pattern, line.lower())
            for word in words:
                total_word_count += 1
                if self.create_dictionary_entry(word):
                    unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def create_dictionary(self, fname):
        total_word_count = 0
        unique_word_count = 0

        with open(fname) as file:
            for line in file:
                # separate by words by non-alphabetical characters
                words = re.findall('[a-z]+', line.lower())
                for word in words:
                    total_word_count += 1
                    if self.create_dictionary_entry(word):
                        unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def get_suggestions(self, string, silent=False):
        """return list of suggested corrections for potentially incorrectly
           spelled word"""
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print("no items in dictionary within maximum edit distance")
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        q_dictionary = {}  # items other than string that we've checked

        while len(queue) > 0:
            q_item = queue[0]  # pop
            queue = queue[1:]

            # early exit
            if ((self.verbose < 2) and (len(suggest_dict) > 0) and
                    ((len(string) - len(q_item)) > min_suggest_len)):
                break

            # process queue item
            if (q_item in self.dictionary) and (q_item not in suggest_dict):
                if self.dictionary[q_item][1] > 0:
                    # word is in dictionary, and is a word from the corpus, and
                    # not already in suggestion list so add to suggestion
                    # dictionary, indexed by the word with value (frequency in
                    # corpus, edit distance)
                    # note q_items that are not the input string are shorter
                    # than input string since only deletes are added (unless
                    # manual dictionary corrections are added)
                    assert len(string) >= len(q_item)
                    suggest_dict[q_item] = (self.dictionary[q_item][1],
                                            len(string) - len(q_item))
                    # early exit
                    if (self.verbose < 2) and (len(string) == len(q_item)):
                        break
                    elif (len(string) - len(q_item)) < min_suggest_len:
                        min_suggest_len = len(string) - len(q_item)

                # the suggested corrections for q_item as stored in
                # dictionary (whether or not q_item itself is a valid word
                # or merely a delete) can be valid corrections
                for sc_item in self.dictionary[q_item][0]:
                    if sc_item not in suggest_dict:

                        # compute edit distance
                        # suggested items should always be longer
                        # (unless manual corrections are added)
                        assert len(sc_item) > len(q_item)

                        # q_items that are not input should be shorter
                        # than original string
                        # (unless manual corrections added)
                        assert len(q_item) <= len(string)

                        if len(q_item) == len(string):
                            assert q_item == string
                            item_dist = len(sc_item) - len(q_item)

                        # item in suggestions list should not be the same as
                        # the string itself
                        assert sc_item != string

                        # calculate edit distance using, for example,
                        # Damerau-Levenshtein distance
                        item_dist = dameraulevenshtein(sc_item, string)

                        # do not add words with greater edit distance if
                        # verbose setting not on
                        if (self.verbose < 2) and (item_dist > min_suggest_len):
                            pass
                        elif item_dist <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (self.dictionary[sc_item][1], item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                        # depending on order words are processed, some words
                        # with different edit distances may be entered into
                        # suggestions; trim suggestion dictionary if verbose
                        # setting not on
                        if self.verbose < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            # now generate deletes (e.g. a substring of string or of a delete)
            # from the queue item
            # as additional items to check -- add to end of queue
            assert len(string) >= len(q_item)

            # do not add words with greater edit distance if verbose setting
            # is not on
            if (self.verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

        # queue is now empty: convert suggestions in dictionary to
        # list for output
        if not silent and self.verbose != 0:
            print("number of possible corrections: %i" % len(suggest_dict))
            print("  edit distance for deletions: %i" % self.max_edit_distance)

        # output option 1
        # sort results by ascending order of edit distance and descending
        # order of frequency
        #     and return list of suggested word corrections only:
        # return sorted(suggest_dict, key = lambda x:
        #               (suggest_dict[x][1], -suggest_dict[x][0]))

        # output option 2
        # return list of suggestions with (correction,
        #                                  (frequency in corpus, edit distance)):
        as_list = suggest_dict.items()
        # outlist = sorted(as_list, key=lambda (term, (freq, dist)): (dist, -freq))
        outlist = sorted(as_list, key=lambda x: (x[1][1], -x[1][0]))

        if self.verbose == 0:
            return outlist[0]
        else:
            return outlist

        '''
        Option 1:
        ['file', 'five', 'fire', 'fine', ...]

        Option 2:
        [('file', (5, 0)),
         ('five', (67, 1)),
         ('fire', (54, 1)),
         ('fine', (17, 1))...]  
        '''

    def best_word(self, s, silent=False):
        try:
            return self.get_suggestions(s, silent)[0]
        except:
            return None


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field, start_time=time()):
        self.field = field
        self.start_time = start_time

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        print(f'[{time()-self.start_time}] select {self.field}')
        dt = dataframe[self.field].dtype
        if is_categorical_dtype(dt):
            return dataframe[self.field].cat.codes[:, None]
        elif is_numeric_dtype(dt):
            return dataframe[self.field][:, None]
        else:
            return dataframe[self.field]


class DropColumnsByDf(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=1, max_df=1.0):
        self.min_df = min_df
        self.max_df = max_df

    def fit(self, X, y=None):
        m = X.tocsc()
        self.nnz_cols = ((m != 0).sum(axis=0) >= self.min_df).A1
        if self.max_df < 1.0:
            max_df = m.shape[0] * self.max_df
            self.nnz_cols = self.nnz_cols & ((m != 0).sum(axis=0) <= max_df).A1
        return self

    def transform(self, X, y=None):
        m = X.tocsc()
        return m[:, self.nnz_cols]


def get_rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))


def split_cat(text):
    try:
        cats = text.split("/")
        return cats[0], cats[1], cats[2], cats[0] + '/' + cats[1]
    except:
        print("no category")
        return 'other', 'other', 'other', 'other/other'


def brands_filling(dataset):
    vc = dataset['brand_name'].value_counts()
    brands = vc[vc > 0].index
    brand_word = r"[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+"

    many_w_brands = brands[brands.str.contains(' ')]
    one_w_brands = brands[~brands.str.contains(' ')]

    ss2 = SymSpell(max_edit_distance=0)
    ss2.create_dictionary_from_arr(many_w_brands, token_pattern=r'.+')

    ss1 = SymSpell(max_edit_distance=0)
    ss1.create_dictionary_from_arr(one_w_brands, token_pattern=r'.+')

    two_words_re = re.compile(r"(?=(\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+))")

    def find_in_str_ss2(row):
        for doc_word in two_words_re.finditer(row):
            print(doc_word)
            suggestion = ss2.best_word(doc_word.group(1), silent=True)
            if suggestion is not None:
                return doc_word.group(1)
        return ''

    def find_in_list_ss1(list):
        for doc_word in list:
            suggestion = ss1.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    def find_in_list_ss2(list):
        for doc_word in list:
            suggestion = ss2.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    print(f"Before empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_name]

    n_desc = dataset[dataset['brand_name'] == '']['item_description'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_desc]

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in n_name]

    desc_lower = dataset[dataset['brand_name'] == '']['item_description'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in desc_lower]

    print(f"After empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    del ss1, ss2
    gc.collect()


def preprocess_regex(dataset, start_time=time()):
    karats_regex = r'(\d)([\s-]?)(karat|karats|carat|carats|kt)([^\w])'
    karats_repl = r'\1k\4'

    unit_regex = r'(\d+)[\s-]([a-z]{2})(\s)'
    unit_repl = r'\1\2\3'

    dataset['name'] = dataset['name'].str.replace(karats_regex, karats_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(karats_regex, karats_repl)
    print(f'[{time() - start_time}] Karats normalized.')

    dataset['name'] = dataset['name'].str.replace(unit_regex, unit_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(unit_regex, unit_repl)
    print(f'[{time() - start_time}] Units glued.')


def preprocess_pandas(train, test, start_time=time()):
    train = train[train.price > 0.0].reset_index(drop=True)
    print('Train shape without zero price: ', train.shape)

    nrow_train = train.shape[0]
    y_train = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, test])

    del train
    del test
    gc.collect()

    merge['has_category'] = (merge['category_name'].notnull()).astype('category')
    print(f'[{time() - start_time}] Has_category filled.')

    merge['category_name'] = merge['category_name'] \
        .fillna('other/other/other') \
        .str.lower() \
        .astype(str)
    merge['general_cat'], merge['subcat_1'], merge['subcat_2'], merge['gen_subcat1'] = \
        zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    print(f'[{time() - start_time}] Split categories completed.')

    merge['has_brand'] = (merge['brand_name'].notnull()).astype('category')
    print(f'[{time() - start_time}] Has_brand filled.')

    merge['gencat_cond'] = merge['general_cat'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_1_cond'] = merge['subcat_1'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_2_cond'] = merge['subcat_2'].map(str) + '_' + merge['item_condition_id'].astype(str)
    print(f'[{time() - start_time}] Categories and item_condition_id concancenated.')

    merge['name'] = merge['name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    merge['brand_name'] = merge['brand_name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    merge['item_description'] = merge['item_description'] \
        .fillna('') \
        .str.lower() \
        .replace(to_replace='No description yet', value='')
    print(f'[{time() - start_time}] Missing filled.')

    preprocess_regex(merge, start_time)

    brands_filling(merge)
    print(f'[{time() - start_time}] Brand name filled.')

    merge['name'] = merge['name'] + ' ' + merge['brand_name']
    print(f'[{time() - start_time}] Name concancenated.')

    merge['item_description'] = merge['item_description'] \
                                + ' ' + merge['name'] \
                                + ' ' + merge['subcat_1'] \
                                + ' ' + merge['subcat_2'] \
                                + ' ' + merge['general_cat'] \
                                + ' ' + merge['brand_name']
    print(f'[{time() - start_time}] Item description concatenated.')

    merge.drop(['price', 'test_id', 'train_id'], axis=1, inplace=True)

    return merge, y_train, nrow_train


def intersect_drop_columns(train: csr_matrix, valid: csr_matrix, min_df=0):
    t = train.tocsc()
    v = valid.tocsc()
    nnz_train = ((t != 0).sum(axis=0) >= min_df).A1
    nnz_valid = ((v != 0).sum(axis=0) >= min_df).A1
    nnz_cols = nnz_train & nnz_valid
    res = t[:, nnz_cols], v[:, nnz_cols]
    return res


def ridge_model(test):
    mp.set_start_method('forkserver', True)

    start_time = time()

    train = pd.read_table(os.path.join(INPUT_PATH, 'train.tsv'),
                          engine='c',
                          dtype={'item_condition_id': 'category',
                                 'shipping': 'category'}
                          )
    
    for col in ['item_condition_id', 'shipping']:
        test[col] = test[col].astype('category')

    '''
    test = pd.read_table(os.path.join(INPUT_PATH, 'test.tsv'),
                         engine='c',
                         dtype={'item_condition_id': 'category',
                                'shipping': 'category'}
                         )
    '''
    print(f'[{time() - start_time}] Finished to load data')
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)

    # submission: pd.DataFrame = test[['test_id']]

    merge, y_train, nrow_train = preprocess_pandas(train, test, start_time)

    meta_params = {'name_ngram': (1, 2),
                   'name_max_f': 75000,
                   'name_min_df': 10,

                   'category_ngram': (2, 3),
                   'category_token': '.+',
                   'category_min_df': 10,

                   'brand_min_df': 10,

                   'desc_ngram': (1, 3),
                   'desc_max_f': 150000,
                   'desc_max_df': 0.5,
                   'desc_min_df': 10}

    stopwords = frozenset(['the', 'a', 'an', 'is', 'it', 'this', ])
    # 'i', 'so', 'its', 'am', 'are'])

    vectorizer = FeatureUnion([
        ('name', Pipeline([
            ('select', ItemSelector('name', start_time=start_time)),
            ('transform', HashingVectorizer(
                ngram_range=(1, 2),
                n_features=2 ** 27,
                norm='l2',
                lowercase=False,
                stop_words=stopwords
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        ('category_name', Pipeline([
            ('select', ItemSelector('category_name', start_time=start_time)),
            ('transform', HashingVectorizer(
                ngram_range=(1, 1),
                token_pattern='.+',
                tokenizer=split_cat,
                n_features=2 ** 27,
                norm='l2',
                lowercase=False
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        ('brand_name', Pipeline([
            ('select', ItemSelector('brand_name', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('gencat_cond', Pipeline([
            ('select', ItemSelector('gencat_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('subcat_1_cond', Pipeline([
            ('select', ItemSelector('subcat_1_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('subcat_2_cond', Pipeline([
            ('select', ItemSelector('subcat_2_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('has_brand', Pipeline([
            ('select', ItemSelector('has_brand', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ])),
        ('shipping', Pipeline([
            ('select', ItemSelector('shipping', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ])),
        ('item_condition_id', Pipeline([
            ('select', ItemSelector('item_condition_id', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ])),
        ('item_description', Pipeline([
            ('select', ItemSelector('item_description', start_time=start_time)),
            ('hash', HashingVectorizer(
                ngram_range=(1, 3),
                n_features=2 ** 27,
                dtype=np.float32,
                norm='l2',
                lowercase=False,
                stop_words=stopwords
            )),
            ('drop_cols', DropColumnsByDf(min_df=2)),
        ]))
    ], n_jobs=1)

    sparse_merge = vectorizer.fit_transform(merge)
    print(f'[{time() - start_time}] Merge vectorized')
    print(sparse_merge.shape)

    tfidf_transformer = TfidfTransformer()

    X = tfidf_transformer.fit_transform(sparse_merge)
    print(f'[{time() - start_time}] TF/IDF completed')

    X_train = X[:nrow_train]
    print(X_train.shape)

    X_test = X[nrow_train:]
    del merge
    del sparse_merge
    del vectorizer
    del tfidf_transformer
    gc.collect()


    X_train, X_test = intersect_drop_columns(X_train, X_test, min_df=1)
    print(f'[{time() - start_time}] Drop only in train or test cols: {X_train.shape[1]}')
    gc.collect()

    
    if develop:
        train_X, valid_X, train_y, valid_y = train_test_split(X_train, y_train, test_size=TEST_SIZE, random_state=SPLIT_SEED)
        
    
    ridge = Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=200, normalize=False, tol=0.01)
    ridge.fit(train_X, train_y)
    print(f'[{time() - start_time}] Train Ridge completed. Iterations: {ridge.n_iter_}')

    
    if develop:
        preds = ridge.predict(valid_X)
        print('->>>>  Ridge dev RMSLE:', rmsle(np.expm1(valid_y), np.expm1(preds))) 
        
    
    predsR = ridge.predict(X_test)
    print(f'[{time() - start_time}] Predict Ridge completed.')

    '''
    submission.loc[:, 'price'] = np.expm1(predsR)
    submission.loc[submission['price'] < 0.0, 'price'] = 0.0
    submission.to_csv("submission_ridge.csv", index=False)
    '''
    print(' Ridge: ...writing the results...')

    print(predsR)
    print('-' * 20)    
    
    return predsR
    
    
#------------------------------------------------------------------------       
#------------------------------------------------------------------------       
#------------------------------------------------------------------------   
def main():
    submission = pd.read_table('../input/test.tsv', engine='c', usecols=['test_id'])
    n_test = len(submission)
    print('test_len: ', n_test)

    # Plan a:
    preds = []
    for df_test in load_test():    
        print('-'*50)
        print('-'*50)
        print('Working on a chunk of test set')
        print('-'*50)
        print('-'*50)
        predsR = ridge_model(df_test)
        predsTF = tf_method(df_test)
        preds.append((np.expm1(predsR) * 0.40 + np.expm1(predsTF) * 0.60).flatten().tolist())

    print('sub len: ', len(preds))    
    submission['price'] = np.array(preds).reshape(-1, 1)
    
    # Plan b:   
    '''
    try: 
        preds = []
        for df_test in load_test():    
            print('-'*50)
            print('-'*50)
            print('Working on a chunk of test set')
            print('-'*50)
            print('-'*50)
            predsR = ridge_model(df_test)
            predsTF = tf_method(df_test)
            preds.append((np.expm1(predsR) * 0.40 + np.expm1(predsTF) * 0.60).flatten().tolist())
        print('sub len: ', len(preds))    
        submission['price'] = np.array(preds).reshape(-1, 1)  
    except:
        submission['price'] = np.array([26.7] * n_test).reshape(-1, 1)
    '''

    # Write to output
    submission.to_csv("submission_ridge_tf.csv", index=False)
        


if __name__ == '__main__':
    main()