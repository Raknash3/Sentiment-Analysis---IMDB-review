import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#the embedding layer takes two arguments
#one the possible no of words in vocabulary
#and the dimensionality of the embeddings, here 32

embedding_layer= layers.Embedding(1000, 32)

#Sentiment classifier on IMDB movie review data

vocab_size = 10000
imdb= keras.datasets.imdb
(train_data,train_label),(test_data,test_label)=imdb.load_data(num_words= vocab_size)
#each data in the train data and test data are already encoded
#since each review can be in different length we use padding 
word_index=imdb.get_word_index()
word_index = {k:(v+3)for k,v in word_index.items()}
word_index['<PAD>']=0
maxlen=500
train_data= keras.preprocessing.sequence.pad_sequences(train_data,value=word_index['<PAD>'],
                                                       padding='post',
                                                        maxlen=maxlen)
test_data= keras.preprocessing.sequence.pad_sequences(test_data,value=word_index['<PAD>'],
                                                       padding='post',
                                                        maxlen=maxlen)
#creating a model
#The first layer is embedding layer, the input is integer encoded strings of same length, i/p shape= 500 and dim =16 
#Global average 1D: second layer returns a fixed length o/p vector for each example by averaging over the sequence dimension
#the next three lines are layers and activation function
embedding_dim=16
model=keras.Sequential([
    layers.Embedding(vocab_size,embedding_dim,input_length=maxlen),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16,activation='relu'),
    layers.Dense(1,activation='sigmoid')
    ])
#compile the data
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#train the model
his=model.fit(
    train_data,train_label,
    epochs=30,
    batch_size=512, 
    validation_data=(test_data,test_label))
    