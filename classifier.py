# Required Python Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import nltk
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder
import pdb
import numpy as np
def score_fn(a,tup,b):
	#print(a)
	if a>0:
		return 1
	return 0

def evaluate(text, pair):
	words=nltk.tokenize.word_tokenize(text)
	bigrams=BigramCollocationFinder.from_words(words)
	#pdb.set_trace()
	return bigrams.score_ngram(score_fn,pair[0],pair[1])

# f=open("bigramsEdited.txt","r")
# dataset=pd.read_csv("./train.csv")
# print(dataset.columns.values)

# lis=f.readlines()[:10]#CHANGE BACK AFTER
# for line in lis:
# 	print(line)
# 	arr=line.split("\t")
# 	tup=eval(arr[0])
# 	print(tup)
# 	dataset[str(tup)]=dataset.apply(axis=1,func=lambda x: 1 if evaluate(x[1],tup)==1 else 0)
# print(dataset)
# dataset.to_pickle("./dataset.txt")

dataset= pd.read_pickle("./dataset.txt")
clf=RandomForestClassifier(n_estimators=20)
X=dataset[dataset.columns.difference(["author","text","id"])]
print(X)
train_x, test_x, train_y, test_y = train_test_split(X, dataset["author"],train_size=0.6)
clf.fit(train_x,train_y)
print(clf.score(train_x,train_y))
print(clf.score(test_x,test_y))
