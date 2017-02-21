import numpy as np
import pandas as pd
import re
import pickle

import common
from common.db_utils import *

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.corpus import stopwords

from gensim.models import Word2Vec


SELECT_DESCRIPS = """ ;"""

def ie_preprocess(document):
	sentences = sent_tokenize(document.lower().replace('\n', '. '))
	sentences = [word_tokenize(sent) for sent in sentences]
	# sentences = [pos_tag(sent) for sent in sentences]
	return sentences

def word_sentence_tokenize(paragraph):
	return [word_tokenize(s) for s in sent_tokenize(paragraph.replace('\n', '. '))]


def pull_tokens_from_pos_tokens(pos_tokens):
	try:
		d = pos_tokens
		val = [x[0] for x in d[0] if x[0] not in stopwords.words()]
	except: 
		val = None
	finally:
		return val


def vectorize_list_of_list_of_tokens(list_of_lists, model):
	vectors = list()
	for alist in list_of_lists:
		for token in alist:
			if token in model.vocab:
				vectors.append(model[token])
	return vectors

def pad_flat_vecs_to_max_len(vec, max_len):
	if len(vec) < max_len:
		padding = [0.] * (max_len - len(vec))
		vec.extend(padding)
	return vec

def save_company_descriptions(df, filename = 'company_vecs.txt'):
	with(open(filename, 'w')) as f:
		for index, row in df[['id_company', 'sentence_vecs_flat']].iterrows():
			if index % 100 == 0:
				print index,
			f.write('%s,%s\n' % (row.id_company, ','.join([str(x) for x in row.sentence_vecs_flat])))


def main():

	conn = init_db()
	df = query_to_df(SELECT_DESCRIPS, conn, ['id_company', 'company', 'descrip'])

	text_col = 'descrip'
	# df['pos_tokens'] = df[text_col].apply(lambda doc: ie_preprocess(doc)
	# df['tokens'] = df.pos_tokens.apply(lambda d: pull_tokens_from_pos_tokens(d))
	df['tokens'] = df[text_col].apply(lambda doc: ie_preprocess(doc))
	df['sentences'] = df.tokens.apply(lambda t: ' '.join(t) if t else None)
	
	sentences = []
	for cv in df[df.tokens.notnull()].tokens.tolist():
		for c in cv:
			sentences.append(c)
	model = Word2Vec(sentences, min_count=3, size = 25)


	df['sentence_vecs'] = df.tokens.apply(lambda tokens: vectorize_list_of_list_of_tokens(tokens, model))
	df['sentence_vecs_flat'] = df.sentence_vecs.apply(lambda vecs: [item for alist in vecs for item in alist])
	max_vecs_len = max(df.sentence_vecs_flat.apply(lambda x: len(x)))
	df['sentence_vecs_padded'] = df.sentence_vecs_flat.apply(lambda vec: pad_flat_vecs_to_max_len(vec, max_vecs_len))

	save_company_descriptions(df, filename = 'company_vecs.txt')
	MODEL_NAME = 'com_descrips.bin'
	model.save_word2vec_format(MODEL_NAME, binary = True)

	com_tags = pd.read_table('com_tags.txt')
	tags = pd.read_table('tagmaster.txt')
	unq_tags = tags.tag.values.tolist()
	tags = tags.merge(com_tags, how = 'inner', on = ['id_tag'])

	for t in unq_tags:
		id_companies = tags.loc[tags.tag == t, 'id_company'].values.tolist()
		df[t] = df.id_company.isin(id_companies)


	tagname = 'fin tech ( fin tech )'
	X_vecs = map(np.array, df.sentence_vecs_padded.values)
	y_vecs = np.array([(1,0) if _ else (0,1) for _ in df[tagname].values])

	PICKLE_NAME = 'fintech_XY.pickle'
	pickle.dump([X_vecs, y_vecs], open(PICKLE_NAME, 'wb'))

