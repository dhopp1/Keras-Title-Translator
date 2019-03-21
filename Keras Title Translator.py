# Databricks notebook source
import pandas as pd
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
data = spark.read.csv('FileStore/tables/cleaned_translations.csv', header = True)
data = data.toPandas()

# COMMAND ----------

def clean_train_test(data, train_perc, x, y, limit = -1):
#function to drop nones, get subset of translation pairs if necessary
#data: pd dataframe of translation pairs, train_perc: percent to train on, remainder will be test set, x: name of source language column, y: name of target language column, limit: number of rows from dataframe to keep, -1 = all
  #limit
  if limit == -1:
    limit = len(data)
  data = data.iloc[:limit,:]
  #drop nones
  data = data.loc[~data.applymap(lambda x: x is None).iloc[:,2],:]
  data = data.loc[~data.applymap(lambda x: x is None).iloc[:,1],:].reset_index(drop=True)
  #train test split
  source_train, source_test, target_train, target_test = train_test_split(data[x], data[y], test_size=(1-train_perc))
  return source_train, source_test, target_train, target_test

# COMMAND ----------

#data prep
#create tokenizer, input pandas series
def create_tokenizer(sentences):
  sentences = list(sentences)
  tokenizer = keras.preprocessing.text.Tokenizer()
  tokenizer.fit_on_texts(lines)
  return tokenizer

#get max sentence length, input pandas series
def max_length(sentences):
  sentences = list(sentences)
  return max(len(line.split()) for line in sentences)

#creating tokenizers, max lengths etc.
source_tokenizer = create_tokenizer(source_train)
target_tokenizer = create_tokenizer(target_train)
source_vocab_size = len(source_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1
source_length = max_length(source_train)
target_length = max_length(target_train)

#encode sequences/pad
def encode_sequences(tokenizer, length, lines):
	#integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	#pad sequences with 0 values
	X = keras.preprocessing.sequence.pad_sequences(X, maxlen=length, padding='post')
	return X

#one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = keras.utils.to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = np.array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

#defining the model
def define_model(source_vocab, target_vocab, source_timesteps, target_timesteps, n_units):
	model = keras.models.Sequential()
	model.add(keras.layers.Embedding(source_vocab, n_units, input_length=source_timesteps, mask_zero=True))
	model.add(keras.layers.LSTM(n_units))
	model.add(keras.layers.RepeatVector(target_timesteps))
	model.add(keras.layers.LSTM(n_units, return_sequences=True))
	model.add(keras.layers.TimeDistributed(keras.layers.Dense(target_vocab, activation='softmax')))
	return model

#model prediction/evaluation
#map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
#generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)
 
#evaluate model
def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# translate encoded source text
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, eng_tokenizer, source)
		raw_target, raw_src = raw_dataset[i]
		if i < 10:
			print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
		actual.append(raw_target.split())
		predicted.append(translation.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# COMMAND ----------

#initial cleaning
raw_source_train, raw_source_test, raw_target_train, raw_target_test = raw_clean_train_test(data, .8, 'english', 'french', 5000)
#prepare training data
train_source = encode_sequences(source_tokenizer, source_length, source_train)
train_target = encode_sequences(target_tokenizer, target_length, target_train)
train_target = encode_output(train_target, english_vocab_size)
#prepare validation data
test_source = encode_sequences(source_tokenizer, source_length, source_test)
test_target = encode_sequences(french_tokenizer, target_length, target_test)
test_target = encode_output(test_target, english_vocab_size)

#create model
model = define_model(source_vocab_size, target_vocab_size, source_length, target_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')

#evaluate model
evaluate_model(model, target_tokenizer, train_source, raw_source_train)
evaluate_model(model, target_tokenizer, test_source, raw_source_test)
