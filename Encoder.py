#Encoder takes a list of token IDs from input_text_processor
#Looks up embedding vector for each token using layers.Embedding
#Processes the embeddings into a new sequence using layers.GRU
#returns the processed sequence to pass to the attnetion head, and the internal state to initialize the decoder

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, units, input_vocab_size, embedding_dim):
        super(Encoder, self).__init__()
        self.units = units
        self.input_vocab_size = input_vocab_size
        
        # Embedding converts our tokens to vectors
        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size, embedding_dim)

        #GRU layers process our vectorized tokens
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')          

    def call(self, tokens, state=None):        

        #The embedding layer looks up the embedding for each token
        vectors = self.embedding(tokens)        

        #3. Gru processes the embedding sequence        
        output, state = self.gru(vectors, initial_state=state)
        #output, state = self.gru(output, initial_state=state) #Uncomment for additional GRU layers
        #output, state = self.gru(output, initial_state=state)
        #output, state = self.gru(output, initial_state=state)        

        # Our output will have a shape: (batch, s, enc_units)
        # Our state will have a shape: (batch, enc_units)

        return output, state

