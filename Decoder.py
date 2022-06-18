import tensorflow as tf
from attention import Attention
from ShapeChecker import ShapeChecker
import typing

from typing import Tuple

class DecoderInput(typing.NamedTuple):
    new_tokens: typing.Any
    enc_output: typing.Any
    mask: typing.Any 

class DecoderOutput(typing.NamedTuple):
    logits: typing.Any
    attention_weights: typing.Any

class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim

        #Step 1: embedding layer converts token IDs to vectors

        self.embedding = tf.keras.layers.Embedding(self.output_vocab_size, embedding_dim)

        #Step 2: RNN Keeps track of previously generated text
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, 
                    recurrent_initializer='glorot_uniform')

        
        #Step 3: RNN output will be query for the attention layer
        self.attention = Attention(self.dec_units)

        #Step 4: Eqn (3) converting ct to at
        self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh, use_bias=False)

        #Step 5. Fully connected layer produces the logits for each output token
        self.fc = tf.keras.layers.Dense(self.output_vocab_size)

    
    def call(self, inputs: DecoderInput, state=None) -> Tuple[DecoderOutput, tf.Tensor]:
        shape_checker = ShapeChecker()
        shape_checker(inputs.new_tokens, ('batch', 't'))
        shape_checker(inputs.enc_output, ('batch', 's', 'enc_units'))
        shape_checker(inputs.mask, ('batch', 's'))

        if state is not None:
            shape_checker(state, ('batch', 'dec_units'))

        #tep 1: Lookup embeddings
        vectors = self.embedding(inputs.new_tokens)
        shape_checker(vectors, ('batch', 't', 'embedding_dim'))

        #Step 2: Process one step with the RNN
        rnn_output, state = self.gru(vectors, initial_state=state)

        shape_checker(rnn_output, ('batch', 't', 'dec_units'))
        shape_checker(state, ('batch', 'dec_units'))

        #Step 3: Use RNN output as query for attetion over the encoder output
        context_vector, attention_weights = self.attention(query=rnn_output, value=inputs.enc_output, mask=inputs.mask)
        shape_checker(context_vector, ('batch', 't', 'dec_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))

        #Step 4: Eqn (3) Join context_vector and rnn_output
        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)

        #Step 4: eqn 3 at = tanh(Wc@[ch:ht])
        attention_vector = self.Wc(context_and_rnn_output)
        shape_checker(attention_vector, ('batch', 't', 'dec_units'))

        #Step 5 generate logic predictions
        logits = self.fc(attention_vector)
        shape_checker(logits, ('batch', 't', 'output_vocab_size'))

        return DecoderOutput(logits, attention_weights), state

