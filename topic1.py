from gensim.models import CoherenceModel

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import json
skip  = 100
print(skip)
data_train = json.load(open('model.json', 'r'))
#data_test = json.load(open('test.json','r'))

#print(type(data_test['captions']))
doc = data_train['captions']
# doc  = [doc1,doc2,doc3,doc4,doc5,doc6]



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

doc_term_matrix = [dictionary.doc2bow(doc1) for doc1 in doc_clean]




Lda = gensim.models.ldamodel.LdaModel

# ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=1)


# doc_lda = ldamodel[doc_term_matrix]
# print(ldamodel.print_topics(num_topics=10, num_words=5))
# print(doc_lda[5])
# print(data_test['captions'][5])


epochs = 50
num = 0
f = open('result2.txt','w')

for i in range(100):
    num = num+skip
    ldamodel = Lda(doc_term_matrix, num_topics=num, id2word = dictionary, passes=epochs)
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=doc_clean, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(coherence_lda,num)
    ldamodel.save('topic_model'+str(i))
    f.write("Coherence " +str(coherence_lda)+" No of topics "+str(num))

f.close()
# print(doc_term_matrix[0])