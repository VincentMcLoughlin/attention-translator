#Encoder takes a list of token IDs from input_text_processor
#Looks up embedding vector for each token using layers.Embedding
#Processes the embeddings into a new sequence using layers.GRU
#returns the processed sequence to pass to the attnetion head, and the internal state to initialize the decoder

import tensorflow as tf
from ShapeChecker import ShapeChecker

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.input_vocab_size = input_vocab_size
        
        # The embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size, embedding_dim)

        #GRU RNN layers processes those vectors sequentially
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, tokens, state=None):
        shape_checker = ShapeChecker()
        shape_checker(tokens, ('batch', 's'))

        # 2. The embedding layer looks up the embedding for each token
        vectors = self.embedding(tokens)
        shape_checker(vectors, ('batch', 's', 'embed_dim'))

        #3. Gru processes the embedding sequence
        # output shape: (batch, s, enc_units)
        # state shape: (batch, enc_units)

        output, state = self.gru(vectors, initial_state=state)
        shape_checker(output, ('batch', 's', 'enc_units'))
        shape_checker(state, ('batch','enc_units'))

        return output, state

