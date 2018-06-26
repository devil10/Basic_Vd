import os
import json
import h5py
import argparse
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm




parser = argparse.ArgumentParser()

parser.add_argument('-input_json_train', default='visdial_0.9_train.json', help='Input `train` json file')



# Output files
parser.add_argument('-output_train_json', default='train.json', help='Output json file')
parser.add_argument('-output_test_json', default='test.json', help='Output hdf5 file')

if __name__ == "__main__":
	args = parser.parse_args()
	data = json.load(open(args.input_json_train, 'r'))
	out = {}
	out['captions'] = []
	out['questions'] = []


	out_test = {}
	out_test['captions'] = []
	out_test['questions'] = []

	for i in range(60000):
		out['captions'].append(data['data']['dialogs'][i]['caption'])
		out['questions'].append(data['data']['questions'][data['data']['dialogs'][i]['dialog'][0]['question']])

	for i in range(60000,70000):
		out_test['captions'].append(data['data']['dialogs'][i]['caption'])
		out_test['questions'].append(data['data']['questions'][data['data']['dialogs'][i]['dialog'][0]['question']])	
    
    #for i in range(10):
    #	print(out['captions'][i],out['questions'][i])

	#print(out['captions'])
	#print(out['questions'])


	# for i in range(10):
	# 	print(out['captions'][i],out['questions'][i])

	json.dump(out, open(args.output_train_json, 'w'))
	json.dump(out_test,open(args.output_test_json,'w'))
