import re
from string import punctuation as PUNCTUATION

import nltk
ngrams = nltk.ngrams
sent_tokenize = nltk.tokenize.sent_tokenize
word_tokenize = nltk.tokenize.word_tokenize
pos_tag = nltk.tag.pos_tag
WNLem = nltk.wordnet.WordNetLemmatizer

CHARMAP = {
	'\xe2\x80\x9c' : '"',
	'\xe2\x80\x9d' : '"',
	'\xe2\x80\x99' : "'",
	'\xe2\x80\x94' : '-',	
	'\xe2\x80\xbe' : ' ',
	'\xff' : ' ',
	'\\n' : ' '
	}

RE_PUNCTUATION = re.compile('[%s]' % PUNCTUATION)
CONTRACTIONS = {  "ain't": "am not; are not; is not; has not; have not", "aren't": "are not; am not", "can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so is", "that'd": "that wouldhad", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
RE_POSSESSIVE_CONTRACTIONS = re.compile('\'\S?')
RE_NUMBER = re.compile('\d*[\.|,]?\d+')
RE_URL = re.compile("(?:(?:https?|ftp)://)(?:\S+(?::\S*)?@)?(?:(?!(?:10|127)(?:\.\d{1,3}){3})(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)(?:\.(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)*(?:\.(?:[a-z\u00a1-\uffff]{2,})))(?::\d{2,5})?(?:/\S*)?")
RE_TWITTER = re.compile("@[a-zA-Z0-9_]{1,15}")

class TextPreprocessing(object):
	"""General Text Preprocesser"""

	charmap = CHARMAP
	ngrams_range = (1,2)
	word_tokenizer = word_tokenize
	sent_tokenizer = sent_tokenize
	stopwords = nltk.corpus.stopwords.words('english')
	stopwords_filepaths = list()
	keywords = list()

	def __init__(self, *args, **kwargs):
		for attr, val in kwargs.iteritems():
			self.__setattr__(attr, val)
			if self.stopwords_filepaths:
				self.load_additional_stopwords()

	def load_additional_stopwords(self):
		new_stopwords = list()
		for f in self.stopwords_filepaths:
			new_stopwords += [str(ln.strip()) for ln in open(f).readlines()]
		self.update_stopwords(new_stopwords)

	def update_stopwords(self, new_stopwords):
		self.stopwords = list(set(self.stopwords + new_stopwords))

	def preprocess(self, text):
		try:
			clean_text = self.process_fulltext(text)
			sentences = sent_tokenize(clean_text)
			processed_text = ' '.join([self.remove_punctuation(s) for s in sentences]).lower()
			
			# processed_text = lemmatize_text(processed_text)
			# processed_text = remove_keywords_from_training(processed_text, self.keywords)
		except Exception as e:
			# print e
			return text
		return processed_text

	def process_fulltext(self, text):
		text = self.map_nonascii_chars(text)
		text = self.compress_whitespace(text)
		text = self.expand_contractions(text)
		text = self.remove_urls(text)
		text = self.remove_twitter_handles(text)
		text = self.remove_numbers(text)
		return text
	
	def map_nonascii_chars(self, text):
		for charfrom, charto in self.charmap.iteritems():
			text = text.replace(charfrom, charto)
		return text

	def compress_whitespace(self, text):
		text = re.sub('\s+', '\x20', text).strip()
		return text

	def expand_contractions(self, text):
		for contraction, expanded in CONTRACTIONS.iteritems():
			text = text.replace(contraction, expanded)
		text = re.sub(RE_POSSESSIVE_CONTRACTIONS, "", text)			
		return text

	def remove_urls(self, text):
		text = re.sub(RE_URL, "__URL__", text)
		return text

	def remove_twitter_handles(self, text):
		text = re.sub(RE_TWITTER, "__HANDLE__", text)
		return text

	def remove_numbers(self, text):
		text = re.sub(RE_NUMBER, "__NUMBER__", text)
		return text

	def remove_punctuation(self, sentence):
		clean_sentence = re.sub(RE_PUNCTUATION, "", sentence)
		if sentence[-1] in PUNCTUATION:
			clean_sentence += sentence[-1]
		return clean_sentence


	def lemmatize_text(self, text, wnl = WNLem()):
		lemmas = list()
		try:
			for word in word_tokenize(text):
				lemmas.append(wnl.lemmatize(word))
			lemma_str = ' '.join(lemmas)
		except UnicodeDecodeError:
				lemma_str = text
		return lemma_str

	def remove_keywords_from_training(self, text, keywords, wnl = WNLem()):
		for kw in keywords:
			kw_lemma = re.compile('\w*' + wnl.lemmatize(kw) + '\w*')
			text = re.sub(kw_lemma, '', text)
			if len(kw.split()) > 1:
				for kw_piece in kw.split():
					kw_piece_lemma = re.compile('\w*' + wnl.lemmatize(kw_piece) + '\w*')
					text = re.sub(kw_piece_lemma, '', text)
		return text

