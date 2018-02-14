# Kaggle-Mercari
    
***

# Ideas/things to do


* Feb 12 - Top matters:
    * Weights of 3 models
    * [FastText](https://www.kaggle.com/nzw0301/simple-keras-fasttext-val-loss-0-31): 1-2 gram
    * [TF dataset](https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428)    

* Feb 13: tune models: [My model improves from 0.433 to 0.410 with these attempts](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/49430#280947):
    * . Scale numerical inputs (if you have any)
    * 2. Add a BatchNormalization layer
    * 3. Try different combinations of activations
    * 4. Change the optimizer (A fine-tuned Adam greatly improves my model!)
    * 5. Change the kernel initializers and bias initializers
    * 6. Try different combinations of batchsize (e.g. an epoch with large batchsize + an epoch with small batchsize)

    
***
# Versions:
* v8:
    * embedding for cat, brand, cond ...
* v7:
    * adam lr
* v6
    * Tf: more epoch (v)
    * FM: 18 ?
    * FTRL: beta=0.01 (v)
    * WB: (no)
    
* v5
    * delete stemmer and add two epoches (v)
    * log price (v)
    * REMEMBER to add fc back (v)
* v4
    * elu instead of relu (v)
    * delete one fc layer (v)
    * delete dropout after fc

* V3 
    * cnn
    * dropout after cnn (v)
    * 2gram for cnn
    * rnn



***

# Worth a read:
* [strategy](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/45291)
* [Ridge: performance/computation time trade off](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/45160)
* [ensemble averaging](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/46568)
* [Why Ridge is much better than other sklearn models](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/46411)
* [Efficient Way to do TFIDF](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/46548)
* [Using log price as Dependent Variable](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/46798) But becarefull with those "without zero price" kernel, as it also remove it from the validation set it makes local CV score useless. If you want to remove zero price,, remove it inside the fold, so the validation set still resemble the original dataset, and then your CV score shall resemble LB
* [Wordbatch(TFIDF) vs WordSequence](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/47504)
* [Best single model](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/47167)
* [Wordbatch for preprocessing and modeling](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/47295)
* [Surpass 0.40000](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/48378)
* [LB shake up](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/48629#277733)
* CNN or RNN: [Best single model](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/47167)


## Top players
* [LB](https://www.kaggle.com/c/mercari-price-suggestion-challenge/leaderboard)
    * [Konstantin Lopuhin](https://www.kaggle.com/lopuhin/discussion?sortBy=latestPost&group=commentsAndTopics&page=1&pageSize=20)
    * [Paweł Jankiewicz](https://www.kaggle.com/paweljankiewicz/discussion?sortBy=latestPost&group=commentsAndTopics&page=1&pageSize=20)
    * [anttip](https://www.kaggle.com/anttip/discussion?sortBy=latestPost&group=commentsAndTopics&page=1&pageSize=20) 
    * [RDizzl3](https://www.kaggle.com/rdizzl3/discussion?sortBy=latestPost&group=commentsAndTopics&page=1&pageSize=20)

    
***

# Useful features

* Len of text
* Mean price of each category
* Mean of brand/shipping
* Average of word embeddings: Lookup all words in Word2vec and take the average of them. [paper](https://www.cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf), [Github](https://github.com/miyyer/dan) [Quora](https://www.quora.com/How-do-I-compute-accurate-sentence-vectors-from-Word2Vec-tool)
* Better way to remove stop word [cached](https://stackoverflow.com/questions/19560498/faster-way-to-remove-stop-words-in-python)
* [Reduce TF time](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/48378#274654)
* Drop price = 0 or < 3 ([link](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/44040), [link](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/45066))


***

# Tricks

* Stage 2: [1](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/43948), [2](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/45212), [Mine](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/49150)

* Rewrite the code: 
    * "without merge(fitting on train and transforming on test) my CV and LB loss increased by 0.009. I can't figure out the reason." [Link](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/47295#278283)
    * Test set into batches. [link](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/47167#271807)
    * Better val set for TF


***

## Tried:

* Combine (condition and shipping)
* [Concatination of brand, item description and product name](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/46381)
* One dimmensionfor item_condition: https://www.kaggle.com/nvhbk16k53/associated-model-rnn-ridge/versions#base=2256015&new=2410057
* Other features for TF: [Quora solutions](https://www.kaggle.com/c/quora-question-pairs/discussion/34325)
    * [No 1](https://www.kaggle.com/c/quora-question-pairs/discussion/34355):  Number of capital letters, question marks etc...
    * [No 3](https://www.kaggle.com/c/quora-question-pairs/discussion/34288): We used TFIDF and LSA distances, word co-occurrence measures (pointwise mutual information), word matching measures, fuzzy word matching measures (edit distance, character ngram distances, etc), LDA, word2vec distances, part of speech and named entity features, and some other minor features. These features were mostly recycled from a previous NLP competition, and were not nearly as essential in this competition.
    * [No 8](https://www.kaggle.com/c/quora-question-pairs/discussion/34371) -> a lot
    * https://www.kaggle.com/shujian/naive-xgboost-v2/versions
    * Tune FM: Compare [1](https://www.kaggle.com/anttip/wordbatch-ftrl-fm-lgb-lbl-0-42555) and [2](https://www.kaggle.com/dromosys/wordbatch-with-fm-ftrl-9ae39c). [topic](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/47295), [kernel](https://www.kaggle.com/anttip/wordbatch-ftrl-fm-lgb-lbl-0-42555). 
* [Ridge tuning](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/46411):
    * RDizzl3: Try playing with these parameters and see if you can get similar results to Ridge: alpha, eta0, power_t and max_iter. I have been able to get within 0.002 of my ridge predictions (validation) and it is faster.
* [Text cleaning](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/46568)
    * RDizzl3: I have created rules for replacing text and even missing brand names that do bring some improvement to my score. 
    * Darragh: I didn't do too much on hand built feature engineering, but have got some boost with working on improving the tokenization. Still looking for what the top guys have done :)  - nltk - ToktokTokenizer
    * [Text norm](https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/46057) 





## Models:

* [Wordbatch FTRL+FM+TF](https://www.kaggle.com/shujian/wordbatch-ftrl-fm-tf?scriptVersionId=2346895): Public Score 0.41803
    * [Wordbatch FTRL+FM+LGB](https://www.kaggle.com/serigne/wordbatch-ftrl-fm-lgb-lb-0-424xx?scriptVersionId=2266455): Public Score 0.42497
    * [Tensorflow starter: conv1d + emb](https://www.kaggle.com/lscoelho/tensorflow-starter-conv1d-emb-0-43839-lb-v08?scriptVersionId=2084098): Public Score 0.43839
* [Wordbatch FTRL+FM+TF+new features](https://www.kaggle.com/shujian/wordbatch-ftrl-fm-tf-new-features?scriptVersionId=2399393): Public Score 0.41502
