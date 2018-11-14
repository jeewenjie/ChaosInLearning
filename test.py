# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 21:32:54 2018

@author: Wen Jie
"""

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import nltk
import tensorflow as tf
from tensorflow import keras

## Define columns
r_cols = ['index','date','result','whiteskill','blackskill','num_move','is_there_date','is_there_result','istherewhiteskill','isthereblackskill','isthere_edate','specificsetup','fen','result2','oyrange','badlen']

## Split the single column into r_cols using ' ' as separator to place them into the respective columns  
df = pd.read_csv(r'C:\Users\Wen Jie\OneDrive\Documents\Academics\Chaos Dynamics\Final Project\allwithfilteredanotationssince1998.txt',sep=' ',names = r_cols, nrows=50001)
df = df.drop(df.index[0:5])
# Resets the index so it counts from 0. Else, due to .drop(), the index starts from 5. 
df = df.reset_index(drop=True)


## Define a new dataframe that is the same as above, without separating into r_cols
## i.e, only have one column named '# #'
moves = pd.read_csv(r'C:\Users\Wen Jie\OneDrive\Documents\Academics\Chaos Dynamics\Final Project\allwithfilteredanotationssince1998.txt',nrows=50000)
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

while count<49996:

      s = final_df.loc[final_df.index[count], 'moves']
      s1 += tokenizer.tokenize(s)
      count += 1

# Keras Tokenizer
t = Tokenizer(num_words=None, filters='!"#$%&()*,/:;<=>?@[\]^_`{|}~', lower=False, char_level=False, oov_token=None)
t.fit_on_texts(s1)


####

final_moves = moves

# t has a dictionary object already thanks to fit_on_tex(). Define a dictionary. 
my_dict = t.word_index

# This is so that to_string is used the text won't get truncated with '...'
pd.set_option('display.max_colwidth', -1)

i = 0
while i < 49996:
    
    # New text for each loop
    words = ((final_moves.iloc[i]).to_string(buf=None, na_rep='NaN', float_format=None, header=True, index=False, length=False, dtype=False, name=False, max_rows=None)).split()
    
    # Join all the words with spaces(' '), using the dictionary substitution if possible
    # get(word,word) is there so won't raise an error if key can't be found in dictionary.
    # What this means is this, 
    # "Give me the value at the key 'word', and if it doesn't exist just give me 'word'"
    final_moves.iloc[i] = [tuple(str(my_dict.get(word, word)) for word in words)]
    i += 1

####

xAxis = final_moves.moves
yAxis = final_df.result 

X_train, X_test, y_train, y_test = train_test_split(xAxis, yAxis, test_size=0.3)

