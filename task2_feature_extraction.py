# -*- coding: utf-8 -*-

from task1_data_process import CorpusReader
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

import pickle
import spacy
import gc

def tokenize(sentence):
    return word_tokenize(sentence)

def lemmatize(word_list, stop_words=True):
    '''
    @param: a list of words
    @param: stop_words (True is to remove stopwords, False otherwise)
    @return: a list of lemmatized words after removing stopwords
    '''
    wordnet_lemmatizer = WordNetLemmatizer()
    
    words = [wordnet_lemmatizer.lemmatize(word, 'n').lower() for word in word_list]
    words = [wordnet_lemmatizer.lemmatize(word, 'a').lower() for word in words]
    words = [wordnet_lemmatizer.lemmatize(word, 'v').lower() for word in words]
    
    if stop_words:
        words = [w for w in words if w not in stopwords.words('english')]
    return words

#Part-of-speech (POS) tag the words to extract POS tag features
def pos_tagger(word_list1, word_list2):
    pos_list1 = pos_tag(word_list1)
    pos_list2 = pos_tag(word_list2)
    
    pos1, pos2 = '', ''
    for word, pos in pos_list1:
        pos1 += pos + ' '
    for word, pos in pos_list2:
        pos2 += pos + ' '
    print(pos1, '\n', pos2)
    subseq = lcs_dp(pos1, pos2)
    print("Their common subsequence:")
    print(subseq)
    return len(subseq.split(" ")) / len(word_list1), len(subseq.split(" ")) / len(word_list2)

#Use spaCy to get dependency parsing tree
def dependency_parser(sent1, sent2, score):
    nlp = spacy.load("en_core_web_sm")
    doc1 = nlp(sent1)
    doc2 = nlp(sent2)
    print("starting dependency parsing...")
    with open("processedData/dependency_parsing.txt", "w") as f:
        f.write("score:\t"+str(score)+"\n")
        print("score: ", score)
        f.write(sent1+"\n")
        print(sent1)
        for token in doc1:
            t = "\t{2}({3}-{4}-{6}, {0}-{5})".format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_, token.i+1, token.head.i+1)
            print(t)
            f.write(t+"\n")
        f.write(sent2+"\n")
        print(sent2)
        for token in doc2:
            t = "\t{2}({3}-{4}-{6}, {0}-{5})".format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_, token.i+1, token.head.i+1)
            print(t)
            f.write(t+"\n")         
        f.write('\n')
    print("dependency parsing finished")
  
#Extract bag of words feature with the help of wordNet
def extract_bow_feature(words1, words2):
    res1 = get_wordnet_overlap(words1, words2)
    print('wordnet overlap(sentence1 --> sentence2):', res1)
    res2 = get_wordnet_overlap(words2, words1)
    print('wordnet overlap(sentence2 --> sentence1):', res2)
    if res1 == 0 or res2 == 0:
        print('feature score:', 0)
        return 0
    else:
        print('feature score:', res1 * res2 / (res1 + res2))
        return res1 * res2 / (res1 + res2)
 
#Get the similarity score of two words using wordNet's synset
def get_wordnet_overlap(words1, words2):
    total = 0
    for w1 in words1:
        if w1 in words2:
            total += 1
        else:
            similarity = []
            for w2 in words2:
                try:
                    value = get_similarity(w1, w2)
                    if value is None:
                        value = 0
                    similarity.append(value)
                except AttributeError:
                    similarity.append(0)
            total += np.max(similarity)
    return total / len(words2)

def get_similarity(w1, w2):
    synsets1 = wordnet.synsets(w1)
    synsets2 = wordnet.synsets(w2)
    sim = 0
    max_sim = 0
    for syn1 in synsets1:
        for syn2 in synsets2:
            sim = wordnet.path_similarity(syn1, syn2)
            if sim is not None:
                if sim > max_sim:
                    max_sim = sim
    return max_sim

def extract_ngram_feature(words1, words2):
    ngram_vec = []
    # Consider three n-grams: unigram, bigram, and trigram
    for n in range(1,4):
        s1_ngrams = []
        s2_ngrams = []
        for i in range(len(words1) - n + 1):
            s1_ngrams.append(words1[i:i+n])
        for i in range(len(words2) - n + 1):
            s2_ngrams.append(words2[i:i+n])
        s1_len = len(s1_ngrams)
        s2_len = len(s2_ngrams)
        print("{}-ngram of sentence 1:".format(n))
        print(s1_ngrams)
        print("{}-ngram of sentence 2:".format(n))
        print(s2_ngrams)
        if s1_len == 0 or s2_len == 0:
            ngram_vec.append(0)
        else:
            common_len = max(1, get_intersection(s1_ngrams, s2_ngrams))
            print("common_len: ", common_len )
            ngram_vec.append(2/(s1_len/common_len + s2_len/common_len))
            print("feature socre: ", 2/(s1_len/common_len + s2_len/common_len))
    return ngram_vec    

    
def lcs_dp(input_x, input_y):
    '''
    @param: input_x
    @param: input_y
    @return: the longest common string of input_x and input_y
    '''
    dp = [([0] * len(input_y)) for i in range(len(input_x))]
    maxlen = maxindex = 0
    for i in range(0, len(input_x)):
        for j in range(0, len(input_y)):
            if input_x[i] == input_y[j]:
                if i != 0 and j != 0:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                if i == 0 or j == 0:
                    dp[i][j] = 1
                if dp[i][j] > maxlen:
                    maxlen = dp[i][j]
                    maxindex = i + 1 - maxlen
                
    return input_x[maxindex:maxindex + maxlen] 
    
def get_intersection(s1, s2):
    '''
    @param: n-gram of sentence 1
    @param: n-gram of sentence 2
    @return: n-gram intersection of two sentences 
    '''
    common = []
    for i in s1:
        if i in s2:
            common.append(i)
    return len(common)   
       
def vectorize(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit(data)
    return X.toarray()

def extract_features(data):
    vectorizer = CountVectorizer()
    s1 = []
    s2 = []
    s = []
    scores = []
    pos_feature = []
    ngram_feature =[]
    num_words_diff = []
    bag_of_words_feature = []
    for line in data:
        s1.append(line[1])
        s2.append(line[2])
        s.append(line[1])
        s.append(line[2])
        #Tokenize the two sentences into words
        word_list1 = tokenize(line[1])
        word_list2 = tokenize(line[2])
        #Lemmatize the words to extract lemmas as features
        words1 = lemmatize(word_list1)
        words2 = lemmatize(word_list2)
        #Part-of-speech (POS) tag the words to extract POS tag features
        pos_feature.append(np.array(pos_tagger(words1, words2)))
        #N-gram feature (unigram, bigram, trigram)
        ngram_feature.append(np.array(extract_ngram_feature(words1, words2)))
        #Get the sentence num_of_words difference as a feature
        num_words_diff.append(abs(len(words1) - len(words2)))
        #Extract bag-of-words feature with the help of wordNet
        bag_of_words_feature.append(extract_bow_feature(words1, words2))
        if len(line) > 3:
            scores.append(float(line[3]))
    
    s_vec = vectorizer.fit_transform(s).toarray()
    s1_vec = vectorizer.transform(s1).toarray()
    s2_vec = vectorizer.transform(s2).toarray()
    tfidf_s1 =[]
    tfidf_s2 = []
    tfidf_s = []
    cosine=[]
    #Extract tf_idf feature
    tfidf = TfidfTransformer()
    for i in range(len(s1)):        
        tfidf_res = tfidf.fit_transform([s1_vec[i], s2_vec[i]]).toarray()
        tfidf_1 = tfidf_res[0]
        tfidf_2 = tfidf_res[1]
        tfidf_s1.append(tfidf_1)
        tfidf_s2.append(tfidf_2)
        tfidf_s.append(tfidf_1)
        tfidf_s.append(tfidf_2)
        cosine.append(cosine_similarity([tfidf_1], [tfidf_2])[0][0])
    
    features = np.c_[cosine, pos_feature, ngram_feature, num_words_diff, bag_of_words_feature] 
    return features, scores
    
if __name__ == '__main__':
    #Load the train, dev, and test files
    reader = CorpusReader()
    reader.train_data = reader.loadFile('data/train-set.txt')
    reader.dev_data = reader.loadFile('data/new-dev-set.txt')
    reader.test_data = reader.loadFile('data/test-set.txt', isTestFile=True)
    
    #Example of dependency parsing
    sent1 = 'Micron has declared its first quarterly profit for three years.'
    sent2 = "Micron's numbers also marked the first quarterly profit in three years for the DRAM manufacturer."
    score = 4
    dependency_parser(sent1, sent2, score)
    
    #Extract all the features
    minMaxScalar = MinMaxScaler()
    train_features, train_scores = extract_features(reader.train_data)
    X_train = minMaxScalar.fit_transform(train_features)
    Y_train = np.array(train_scores)
    with open("processedData/train_data.pickle", "wb") as f:
        pickle.dump([X_train, Y_train], f)
        
    dev_features, dev_scores = extract_features(reader.dev_data)
    X_dev = minMaxScalar.transform(dev_features)
    Y_dev = np.array(dev_scores)
    with open("processedData/dev_data.pickle", "wb") as f:
        pickle.dump([X_dev, Y_dev], f)
        
    test_features, _ = extract_features(reader.test_data)
    X_test = minMaxScalar.transform(test_features)
    
    #Get all the ids in the test set as we need ids to output the predictions
    test_ids = []
    for line in reader.test_data:
        test_ids.append(line[0])            
    test_ids = np.array(test_ids)
    with open("processedData/test_data.pickle", "wb") as f:
        pickle.dump([X_test, test_ids], f) 
   
    dev_ids = []
    for line in reader.dev_data:
        dev_ids.append(line[0])            
    dev_ids = np.array(dev_ids)
    with open("processedData/dev_ids.pickle", "wb") as f:
        pickle.dump(dev_ids, f) 
    
    