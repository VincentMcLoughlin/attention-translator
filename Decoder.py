import tensorflow as tf
from attention import Attention
import typing
from typing import Tuple

class DecoderInput(typing.NamedTuple):
    new_tokens: typing.Any
    encoder_output: typing.Any
    mask: typing.Any 

class DecoderOutput(typing.NamedTuple):
    logits: typing.Any
    attention_weights: typing.Any

class Decoder(tf.keras.layers.Layer):
    def __init__(self, units, output_vocab_size, embedding_size):
        super(Decoder, self).__init__()
        self.units = units
        self.output_vocab_size = output_vocab_size
        self.embedding_size = embedding_size

        # embedding layer converts token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(self.output_vocab_size, embedding_size)

        #GRU tracks previously generated text
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')        
        
        #RNN output is attention input
        self.attention = Attention(self.units)

        #Step 4: Wc converts our context vector ct to our attention vector at
        self.Wc = tf.keras.layers.Dense(units, activation=tf.math.tanh, use_bias=False)

        #Step 5. Fully connected layer produces the logits for each output token
        self.fc = tf.keras.layers.Dense(self.output_vocab_size)

    
    def call(self, inputs: DecoderInput, state=None) -> Tuple[DecoderOutput, tf.Tensor]:                

        #Get Embeddings
        vectors = self.embedding(inputs.new_tokens)

        #Step 2: Process one step with the RNN
        gru_hidden_state_output, gru_state = self.gru(vectors, initial_state=state)
        #gru_hidden_state_output, state = self.gru(gru_hidden_state_output, initial_state=state) #Uncomment for additional gru layers
        #gru_hidden_state_output, state = self.gru(gru_hidden_state_output, initial_state=state)
        #gru_hidden_state_output, state = self.gru(gru_hidden_state_output, initial_state=state)
        
        #query is h_t from our equations, encoder_output is our h_s, input it to our attention layer
        context_vector, attention_weights = self.attention(query=gru_hidden_state_output, value=inputs.encoder_output, mask=inputs.mask)        

        #c_t;h_t from our paper, concatenating context and rnn output
        context_and_gru_output = tf.concat([context_vector, gru_hidden_state_output], axis=-1)
        #context_and_gru_output = gru_hidden_state_output #Uncomment to disable attention

        # Get hidden attentional vector h_t =  tanh(Wc[ch:ht])
        attention_vector = self.Wc(context_and_gru_output)        
    
        logits = self.fc(attention_vector)        

        return DecoderOutput(logits, attention_weights), state

