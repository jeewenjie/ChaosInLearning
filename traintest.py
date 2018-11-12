# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:35:55 2018

@author: Wen Jie
"""
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import hashing_trick
from sklearn.model_selection import train_test_split
import nltk

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


placeholder_final_df = final_df 
yAxis = final_df.result 
xAxis = final_df.moves
# 'index=False'  makes index not included in the final string.
xAxis2 = final_df.moves.to_string(buf=None, na_rep='NaN', float_format=None, header=True, index=False, length=False, dtype=False, name=False, max_rows=None)

tokenizer = nltk.RegexpTokenizer('\s+', gaps=True)

tokenizer.tokenize(xAxis2)

t = Tokenizer(num_words=None, filters='!"#$%&()*+,-/:;<=>?@[\]^_`{|}~', lower=False, split=' ', char_level=False, oov_token=None)
t.fit_on_texts(xAxis2)
print(t.word_index)


count = 0 

while count < 49996:
      # hashing_trick requires n input
      n = len(xAxis[count])
      
      # Hashes the tokens into integers. But this method doesn't take into
      # account similar moves in different rows, making them different integer for same moves
      #xAxis[count] = hashing_trick(xAxis[count], n, hash_function=None, filters='!"#$%&()*+,-/:;<=>?@[\]^_`{|}~', lower=False, split=' ')
      
      count += 1

X_train, X_test, y_train, y_test = train_test_split(xAxis, yAxis, test_size=0.3)