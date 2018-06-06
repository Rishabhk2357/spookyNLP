lis=["HPL","MWS","EAP"]
import pandas as pd
import nltk
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder
import pdb
import sklearn
from sklearn.model_selection import train_test_split
train=pd.read_csv("./train.csv")
def score_fn(a,tup,b):
	if a>0:
		return 1
def evaluateConjunction(row):
	words=nltk.tokenize.word_tokenize(row[1])
	tags=nltk.pos_tag(words)
	count=0
	for word in tags:
		if len(word[1])>1 and word[1][:2]=="CC":
			count+=1

	return count/len(words)
def evaluateAdjective(row):
	words=nltk.tokenize.word_tokenize(row[1])
	tags=nltk.pos_tag(words)
	count=0
	for word in tags:
		if len(word[1])>1 and word[1][:2]=="JJ":
			count+=1
	return count/len(words)
def evaluateAdverb(row):
	words=nltk.tokenize.word_tokenize(row[1])
	tags=nltk.pos_tag(words)
	count=0
	for word in tags:
		if len(word[1])>1 and word[1][:2]=="RB":
			count+=1
	return count/len(words)
def evaluateNoun(row):
	words=nltk.tokenize.word_tokenize(row[1])
	tags=nltk.pos_tag(words)
	count=0
	for word in tags:
		if word[1][:1]=="N" or word[1]=="PRP":
			count+=1
	return count/len(words)
def evaluateConj(row):
	words=nltk.tokenize.word_tokenize(row[1])
	tags=nltk.pos_tag(words)
	count=0
	for word in tags:
		if word[1][:1]=="N" or word[1]=="PRP":
			count+=1
	return count/len(words)
def evaluateWords(row):
	words=nltk.tokenize.word_tokenize(row[1])
	return len(words)
def wordLength(row):
	tot=0
	words=nltk.tokenize.word_tokenize(row[1])
	for word in words:
		tot+=len(word)
	return tot/len(words)
f=open("bigrams.txt",'w')
train["adjCount"]=train.apply(evaluateAdjective,axis=1)
train["conjCount"]=train.apply(evaluateConjunction,axis=1)
train["advCount"]=train.apply(evaluateAdverb,axis=1)
train["wordCount"]=train.apply(evaluateWords,axis=1)
train["wordLen"]=train.apply(wordLength,axis=1)
train["nounCount"]=train.apply(evaluateNoun,axis=1)
train.to_pickle("svm.txt")

# train=pd.read_pickle("./svm.txt")
# for auth in lis:
# 	ser=train.groupby("author").get_group(auth)["nounCount"]
# 	print(auth+" "+str(ser.mean())+" "+str(ser.std()))
clf=sklearn.svm.SVC()
del train["id"]
del train["text"]
train_x, test_x, train_y, test_y = train_test_split(train[train.columns.difference(["author"])], train["author"],train_size=0.6)
print(train_x)
print(train_y)
clf.fit(train_x.as_matrix(),train_y.as_matrix())
print(clf.score(test_x.as_matrix(),test_y.as_matrix()))

# for key in lis:
# 	print(key)
# 	f.write(key+"\n")
# 	s=""
# 	bigram_measures = nltk.collocations.BigramAssocMeasures()
# 	for row in train.itertuples():
# 		#print(row)]
# 		if row[3]==key:
# 			s+=(" "+row[2])
# 	words=nltk.tokenize.word_tokenize(s)
# 	fdist=FreqDist(nltk.tokenize.word_tokenize(s))
# 	#print(fdist.most_common(50))
# 	bigrams=BigramCollocationFinder.from_words(words)
# 	bigrams.apply_word_filter(lambda w: w[0].isupper() or not w[0].isalpha())
# 	bigrams.apply_freq_filter(100)
# 	pops=bigrams.score_ngrams(bigram_measures.pmi)
# 	print(pops)

# 	for pair in pops:
# 		#pdb.set_trace()
# 		dic={"HPL":0,"MWS":0,"EAP":0}
# 		for row in train.itertuples():
# 			#print(row)]
# 			words=nltk.tokenize.word_tokenize(row[2])
# 			bigrams_row=BigramCollocationFinder.from_words(words)
# 			if (bigrams_row.score_ngram(score_fn,pair[0][0],pair[0][1])==1):
# 				dic[row[3]]+=1
# 		#print(dic)
# 		if(dic[key]==max(dic.values())):
# 			f.write(str(pair[0]))
# 			f.write("\t")
# 			f.write(str(dic))
# 			f.write("\n")
f.close()

