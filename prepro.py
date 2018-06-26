import os
import json
import h5py
import argparse
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import gensim
from gensim.models import Word2Vec
import numpy as np
import re
#impo



parser = argparse.ArgumentParser()

# Input files
parser.add_argument('-input_json_train', default='train.json', help='Input `train` json file')
parser.add_argument('-input_json_test', default='test.json', help='Input `train` json file')


# Output files
parser.add_argument('-output_json', default='vocab.json', help='Output json file')
parser.add_argument('-output_h5', default='out1.h5', help='Output hdf5 file')

# Options
parser.add_argument('-max_ques_len', default=10, type=int, help='Max length of questions')
parser.add_argument('-max_ans_len', default=20, type=int, help='Max length of answers')
parser.add_argument('-max_cap_len', default=15, type=int, help='Max length of captions')
parser.add_argument('-word_count_threshold', default=5, type=int, help='Min threshold of word count to include in vocabulary')


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n' and i!=','];





def tokenize_data(data1,data2):
    print(" Tokenizing captions...")
    data1['caption_tokens'] = []
    word_counts = {}
    for i, caption in enumerate((data1['captions'])):
        data1['caption_tokens'].append(tokenize(caption))

    print("Tokenizing questions")
    data1['question_tokens'] = []
    for q in (data1['questions']):
        data1['question_tokens'].append(tokenize(q + '?'))



    print(" Tokenizing captions...")
    data2['caption_tokens'] = []
    for i, caption in enumerate((data2['captions'])):
        data2['caption_tokens'].append(tokenize(caption))

    print("Tokenizing questions")
    data2['question_tokens'] = []
    for q in (data2['questions']):
        data2['question_tokens'].append(tokenize(q + '?'))


    for q in data1['question_tokens']:
        for word in q:
            word_counts[word] = word_counts.get(word, 0) + 1

    for q in data2['question_tokens']:
        for word in q:
            word_counts[word] = word_counts.get(word, 0) + 1        

    return data1,data2, word_counts



def encode_vocab(data,word2ind):
    data['question_tokens_encoded'] = []
    for i, q in enumerate(tqdm(data['question_tokens'])):
        data['question_tokens_encoded'].append([word2ind.get(word, word2ind['<UNK>']) for word in q])
    return data 
    

def word2vec(data):
    data['encoded_captions']=[]
    for i , caption in enumerate(data['caption_tokens']):
        vec = np.zeros((args.max_cap_len,300))
        #print(caption)
        for j , word in enumerate(caption):
            if j < args.max_cap_len:
                #print(j)
                try:
                    vec[j,:] = model[word]
                    #if i == 0:
                    #print(vec[j,:])
                    #print(vec[j,:])
                except KeyError:
                    #print('zzzz')
                    #print(word)
                    vec[j,:] = np.zeros((1,300)) 
        #if i == 0:
            #print(vec.reshape(1,4500)[0][300:400])

        data['encoded_captions'].append(vec.reshape(1,4500)[0])
    #print(data['encoded_captions'][0][300:400])    
    return data     

def create_data(data1,data2,word2ind):
    data_mats = {}
    data_mats['train_caption_embeddings'] = data1['encoded_captions']
    data_mats['test_caption_embeddings'] =data2['encoded_captions']
    data_mats['train_questions'] = data1['question_tokens_encoded']
    data_mats['test_questions'] = data2['question_tokens_encoded']


    data_mats['train_questions'] =0*np.ones([len(data1['captions']),args.max_ques_len])
    data_mats['test_questions'] =0*np.ones([len(data2['captions']),args.max_ques_len])


    for i, question in enumerate(data1['question_tokens_encoded']):
        question_len = len(question[0:args.max_ques_len])
        data_mats['train_questions'][i][0:question_len] = question[0:args.max_ques_len]
    
    for i, question in enumerate(data2['question_tokens_encoded']):
        question_len = len(question[0:args.max_ques_len])
        #print(question_len)
        data_mats['test_questions'][i][0:question_len] = question[0:args.max_ques_len]
        #print(data_mats[])
    return data_mats    



if __name__ == "__main__":

    args = parser.parse_args()
    model = Word2Vec.load_word2vec_format('../google.bin', binary=True)
    #print(model['table'])
    data_train = json.load(open(args.input_json_train, 'r'))
    data_test = json.load(open(args.input_json_test,'r'))

    data_train,data_test,word_counts = tokenize_data(data_train, data_test)

    word_counts['<UNK>'] = args.word_count_threshold
    #word_counts['.'] = args.word_count_threshold
    word_counts['<START>'] = args.word_count_threshold
    word_counts['<END>'] = args.word_count_threshold

    vocab = [word for word in word_counts \
    if word_counts[word] >= args.word_count_threshold]
    #print('Words: %d' % len(vocab))
    word2ind = {word: word_ind + 1 for word_ind, word in enumerate(vocab)}
    ind2word = {word_ind: word for word, word_ind in word2ind.items()}

    data_train = encode_vocab(data_train,word2ind)
    data_test = encode_vocab(data_test,word2ind)
    print(data_test['question_tokens_encoded'][0])
    #print(len(data_train['caption_tokens']))
    data_train = word2vec(data_train)
    data_test = word2vec(data_test)
    #print(data_train['encoded_captions'][0][300:400])
    data_mats = create_data(data_train,data_test,word2ind)
    #print(data_mats['train_caption_embeddings'][0][300:400])

    #print(data_test['question_tokens_encoded'][0])
    #print(data_test['caption_tokens'])

    out = {}
    out['ind2word'] = ind2word
    out['word2ind'] = word2ind
    #out['train_questions'] = data_mats['train_questions']
    #out['test_questions'] = data_mats['test_questions']
    out['train_questions_tokens']  = data_train['question_tokens']
    out['test_questions_tokens']  = data_test['question_tokens']
    
    json.dump(out, open(args.output_json, 'w'))



    print('Saving hdf5...')
    f = h5py.File(args.output_h5, 'w')
    
    f.create_dataset('train_caption_embeddings',dtype = 'float32',data = data_mats['train_caption_embeddings'])
    f.create_dataset('test_caption_embeddings',dtype = 'float32',data = data_mats['test_caption_embeddings'])
    f.create_dataset('train_questions',dtype = 'uint32',data = data_mats['train_questions'])
    f.create_dataset('test_questions',dtype = 'uint32',data = data_mats['test_questions'])
    
    # for key in data_mats:
    #     f.create_dataset(key, dtype='float32', data=data_mats[key])
    #     f.create_dataset()
    f.close()


    # filename = args.output_h5
    # f = h5py.File(filename, 'r')
    #print(f['train_caption_embeddings'][0][300:400])
    # List all groups
    
    # Get the data
    #datax = list(f[a_group_key])
    #print(datax[0])

    



    #print(word_counts)
