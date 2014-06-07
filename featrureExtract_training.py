import os
import re
import csv
import nltk
import enchant
import numpy as np
from random import shuffle
from bs4 import BeautifulSoup as bs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation


def getcommentdata():
	print 'getting data...'
	#TODO:Retrive from SQL 

	content = [row[2:4] for row in csv.reader(open("spamclass/data/review_comments_labelled.csv","rb"),delimiter=',',quotechar='"')][1:]+[row[2:4] for row in csv.reader(open("spamclass/data/photo_comments_labelled.csv","rb"),delimiter=',',quotechar='"')][1:]
	shuffle(content)
	X = [r[0] for r in content]
	y = [int(r[1]) for r in content]
	
	return[X, y]



def rules(pred,testx):

	if len(testx.split(' ')) == 1 and 'thank' in testx.lower():
		pred = 0
	if 'facebook' in testx.lower() or 'twitter' in testx.lower() or 'instagram' in testx.lower():
		pred = 1
	if len(testx.split(' ')) <= 5 and ('nice' in testx.lower() or 'good' in testx.lower() or 'amazing' in testx.lower() or 'awesome' in testx.lower()):
		pred = 0
	for emo in [':)',':(',':D',':/','=D']:
		if emo in testx:
			pred = 0
			break
	if len(testx.split(' ')) == 1 and not(testx.isalpha()):
		pred = 1

	d = enchant.Dict("en_US")
	if len(testx.split(' ')) == 1 and d.check(testx.lower()):
		pred = 0

	
	return pred


def sequenceTokens(reg, con):

	token = [i for i in re.findall(reg,con) if i.replace(' ','') != '']
	num_token = len(token)
	seq_token = [i for i in re.findall(reg+'+',con) if i.replace(' ','') != '']
	if seq_token:
		max_token = len(max(seq_token, key=len).replace(' ',''))
	else:
		if token:
			max_token = len(max(token, key=len).replace(' ',''))
		else:
			max_token = 0

	return [num_token, max_token]


def morefeatures(content):

	features = []
	for con in content:
		tot_length = len(con)
		sentences = nltk.sent_tokenize(con)
		num_sentences = len(sentences)
		words = [nltk.word_tokenize(re.sub(r'[^A-Za-z0-9]','',sentence)) for sentence in sentences]
		num_words = len(words)
		avg_len_sent = float(sum([len(sentence) for sentence in sentences]))/float(num_sentences)
		avg_len_word = float(sum([len(word) for word in words]))/float(num_words)

		num_punc, max_punc = sequenceTokens('\W',con)
		num_digit, max_digit = sequenceTokens('\d',con)
		num_cap, max_cap = sequenceTokens('[A-Z]',con)

		newline = len(re.findall('\n',con)) 

		urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', con)
		num_urls = len(urls)
		flag_url = 0
		if urls:
			for url in urls:
				if 'zomato.com' not in url:
					flag_url = 1
					break

		features.append([tot_length, num_sentences, num_words, avg_len_sent, avg_len_word, num_punc, max_punc, num_digit, max_digit, num_cap, max_cap, newline,num_urls, flag_url])

	return features


def train_vectorize(X):

	print 'vectorizing...'
	
	count_vectorizer = CountVectorizer(min_df=1,stop_words='english',ngram_range=(1,3))
	vectors = count_vectorizer.fit_transform(X)

	return [count_vectorizer, vectors]

def fit_vectorize(X,model):
	return model.transform(X)

def addfeatures(X, X_vectors, yes):

	print 'getting more features...'
	if yes == 1:
		matrix = X_vectors.todense()
		get_features = np.array(morefeatures(X))
		features = np.hstack((matrix, get_features))
	else:		
		features = np.array(train_vector.todense())

	return features

def cv_scores(clf, X_features, y, cv):

	count_vec, X_vectors = train_vectorize(X)
	X_features = addfeatures(X, X_vectors,yes = 1)

	return cross_validation.cross_val_score(clf, X_features, np.array(y), cv=cv).mean()

def getTrainTest(X, y):
	if isinstance(X,list):
		size = len(X)
	else:
		size = X.shape[0]
	trainlen = int(size * 0.9)

	return [X[:trainlen], y[:trainlen], X[trainlen:], y[trainlen:]]

def evaluate(predictions, testx, testy):

	results = []
	for i,p in enumerate(predictions):
		predictions[i] = rules(predictions[i],testx[i])
		results.append([str(predictions[i]),str(testy[i]),testx[i]])
		print str(predictions[i]),str(testy[i]),testx[i]
	print len(testx),(predictions!=testy).sum()

	return results

def predict(clf, trainx, trainy, testx):

	clf.fit(trainx, trainy)
	# return clf
	predictions = clf.predict(testx)

	return predictions

if __name__ == "__main__":
	
	X, y = getcommentdata()
	
	# trainx = X
	#train:test :: 8:2
	trainx, trainy, testx, testy = getTrainTest(X,y)

	#creates Bag of Words, bigrams, trigrams
	count_vec, train_vectors = train_vectorize(trainx)
	test_vectors = fit_vectorize(testx, count_vec)

	#Adds structure features
	train_features = addfeatures(trainx, train_vectors,yes = 1)
	test_features = addfeatures(testx, test_vectors,yes = 1)
	
	
	#declaring classifiers
	clf = svm.LinearSVC()	
	clf2 = RandomForestClassifier(n_estimators=20)

	
	#Accuracy by k fold cross validation
	#If choosing this mehtod of evaluation, make sure X and y are retured values from addfeatures(). Also, dividing the data is not necesssary.
	# scores = cv_scores(clf, X, y, cv=5)
	
	clf.fit(train_features, trainy)
	clf2.fit(train_features, trainy)

	predictions = predict(clf, train_features, trainy, test_features)

	results_data = evaluate(predictions, testx, testy)

	joblib.dump(count_vec, 'vectorizer.pkl', compress=9)
	joblib.dump(clf, 'linearsvm.pkl', compress=9)
	joblib.dump(clf2, 'random_forest.pkl', compress=9)


