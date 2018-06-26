from gensim.models import CoherenceModel

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import json
import numpy as np
import h5py



data_train = json.load(open('model.json', 'r'))
#data_test = json.load(open('test.json','r'))

#print(type(data_test['captions']))
doc = data_train['captions']
print(len(doc))

train_size = 60000
test_size = 10000
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()



def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


doc_clean = [clean(doc).split() for doc in doc]   

# print(doc_clean[0])
dictionary = corpora.Dictionary(doc_clean)
# print(dictionary[3])

bow = [dictionary.doc2bow(doc1) for doc1 in doc_clean]


Lda = gensim.models.ldamodel.LdaModel
model = Lda.load('topic_models_trained/topic_model0')
#print(model.get_document_topics(bow[0],minimum_probability = 0))
a = model.get_topic_terms(0,10739)
print(len(a))

vocab_size = len(dictionary)
no_topics = 100


doc_to_topics = np.zeros([train_size+test_size,no_topics])

for i in range(train_size+test_size):
	l = model.get_document_topics(bow[i],minimum_probability = 0)
	for j in range(len(l)):
		(a,b) = l[j]
		doc_to_topics[i][a] =b


topics_to_vocab = np.zeros([no_topics,vocab_size])

for i in range(no_topics):
	l = model.get_topic_terms(i,vocab_size)
	for j in range(len(l)):
		(a,b) = l[j]
		topics_to_vocab[i][a] = b

top_topics = 10

dat_train = np.zeros([train_size,vocab_size])

for i in range(train_size):
	indices = doc_to_topics[i].argsort()[-top_topics:][::-1]
	#print(indices)
	summ = 0
	for j in range(top_topics):
		#print(topics_to_vocab[indices[j]].shape,dat_train[i].shape,type(int(doc_to_topics[i][indices[j]])))
		#print(doc_to_topics[i][indices[j]],indices[j])
		dat_train[i]  = dat_train[i]+doc_to_topics[i][indices[j]]*topics_to_vocab[indices[j]]
		summ =summ+doc_to_topics[i][indices[j]]
	dat_train[i] = dat_train[i]/summ


dat_test = np.zeros([test_size,vocab_size])

for i in range(test_size):
	indices = doc_to_topics[train_size+i].argsort()[-top_topics:][::-1]
	#print(indices)
	summ = 0
	for j in range(top_topics):
		#print(topics_to_vocab[indices[j]].shape,dat_train[i].shape,type(int(doc_to_topics[i][indices[j]])))
		dat_test[i]  = dat_test[i]+doc_to_topics[i+train_size][indices[j]]*topics_to_vocab[indices[j]]
		summ =summ+doc_to_topics[i+train_size][indices[j]]
	dat_test[i] = dat_test[i]/summ
		

#print(doc_to_topics[0])

#print(dat_train[0])
#print(model.get_document_topics(bow[0],minimum_probability = 0))

#print(model.get_topic_terms(0,10))

#print(topics_to_vocab[0])
print(np.sum(dat_train[0]))

f = h5py.File('topics.h5', 'w')
    
f.create_dataset('train_caption_embeddings',dtype = 'float32',data = dat_train)
f.create_dataset('test_caption_embeddings',dtype = 'float32',data = dat_test)
#f.create_dataset('doc_topics',dtype = 'float32',data = doc_to_topics)
#f.create_dataset('topics_to_vocab',dtype = 'float32',data = topics_to_vocab)

#f.create_dataset('test_caption_embeddings',dtype = 'float32',data = data_mats['test_caption_embeddings'])