# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 21:32:54 2018

@author: Wen Jie
"""

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import nltk
from keras.preprocessing import sequence
from keras.layers import Dense, Activation,Dropout
from keras.utils import to_categorical
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

take_n_rows = 75000

## Define columns
r_cols = ['index','date','result','whiteskill','blackskill','num_move','is_there_date','is_there_result','istherewhiteskill','isthereblackskill','isthere_edate','specificsetup','fen','result2','oyrange','badlen']

## Split the single column into r_cols using ' ' as separator to place them into the respective columns  
df = pd.read_csv(r'C:\Users\Wen Jie\OneDrive\Documents\Academics\Chaos Dynamics\Final Project\allwithfilteredanotationssince1998.txt',sep=' ',names = r_cols, nrows=take_n_rows)
df = df.drop(df.index[0:5])
# Resets the index so it counts from 0. Else, due to .drop(), the index starts from 5. 
df = df.reset_index(drop=True)


## Define a new dataframe that is the same as above, without separating into r_cols
## i.e, only have one column named '# #'
moves = pd.read_csv(r'C:\Users\Wen Jie\OneDrive\Documents\Academics\Chaos Dynamics\Final Project\allwithfilteredanotationssince1998.txt',nrows=take_n_rows-1)
moves = moves.drop(moves.index[0:4]) # Same reason as previous section 
# Resets the index so it counts from 0. Else, due to .drop(), the index starts from 4. 
moves = moves.reset_index(drop=True)
## Split the column '# #' into 2 columns('# #' and 'moves') after the '###' token. 
## str[1] takes everything after the token and place into column 'moves'. 
## str[0] takes everything before the token and place into column 'moves'.
moves['moves'] = moves['# #'].str.split('###').str[1] 


# axis = 1 refers to column; 0 to row. 
# np.split divides the dataframe into 2 dataframe. I'm only taking 1 of the 2.  
moves = np.split(moves, [1], axis=1)[1]


final_df = pd.concat([df, moves], axis=1)

##IGNORE##
#traintest_cutoff = int(np.ceil(0.7*50000))
#train_ctrl,train_output = final_df.moves[:traintest_cutoff],final_df.result[:traintest_cutoff]
#test_ctrl, test_output  = final_df.moves[traintest_cutoff:],final_df.result[traintest_cutoff:]


count = 0 

# nltk tokenizer
tokenizer = nltk.RegexpTokenizer('\s+', gaps=True)
s1 = []

while count<(take_n_rows-5):

      s = final_df.loc[final_df.index[count], 'moves']
      s1 += tokenizer.tokenize(s)
      count += 1

# Keras Tokenizer
# Don't filter =,+,.,-,#
t = Tokenizer(num_words=None, filters='!"$%&()*,/:;<>?@[\]^_`{|}~', lower=False, char_level=False, oov_token=None)
t.fit_on_texts(s1)


####

final_moves = moves

# t has a dictionary object already thanks to fit_on_tex(). Define a dictionary. 
my_dict = t.word_index

# This is so that to_string is used the text won't get truncated with '...'
pd.set_option('display.max_colwidth', -1)

i = 0
while i<(take_n_rows-5):
    
    # New text for each loop
    words = ((final_moves.iloc[i]).to_string(buf=None, na_rep='NaN', float_format=None, header=True, index=False, length=False, dtype=False, name=False, max_rows=None)).split()
    
    # Join all the words with spaces(' '), using the dictionary substitution if possible
    # get(word,word) is there so won't raise an error if key can't be found in dictionary.
    # What this means is this, 
    # "Give me the value at the key 'word', and if it doesn't exist just give me 'word'"
    final_moves.iloc[i] = [tuple(str(my_dict.get(word, word)) for word in words)]
    # White win = 1, Black win = 2, Draw = 0
    if final_df.iloc[i,2] == '1-0':
       final_df.iloc[i,2] = 1
    elif final_df.iloc[i,2] == '0-1':
        final_df.iloc[i,2] = 2
    elif final_df.iloc[i,2] == '1/2-1/2':
        final_df.iloc[i,2] = 0
        
    i += 1

####

xAxis = final_moves.moves
yAxis = final_df.result

# Pads the sequence so all have same number(327) of elements in sequence 
xAxis =  sequence.pad_sequences(xAxis, padding = 'post',maxlen = 324)

X_train, X_test, y_train, y_test = train_test_split(xAxis, yAxis, test_size=0.3)

# Normalize X_data
norm = max(map(max, X_test))
X_train = X_train/norm
X_test = X_test/norm

y_train2 = y_train #For testing, delete later
y_test2 = y_test # For testing, delete later

y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

model = Sequential()
#512 -> 16
model.add(Dense(16, input_shape=(324,)))
model.add(Activation('relu'))                            
#model.add(Dropout(0.2))
model.add(Dropout(0.2))
#512 -> 16
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.2))
#model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation('softmax'))
#model.add(Dense(3))
#model.add(Activation('sigmoid'))

# categorical_crossentropy for multi-label 
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X_train, y_train,
          batch_size=128, epochs=20,
          verbose=2,
          validation_data=(X_test, y_test))

# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()