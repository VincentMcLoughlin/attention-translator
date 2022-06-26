import tensorflow as tf
import numpy as np
from Decoder import Decoder, DecoderInput

class Translator(tf.Module):

    def __init__(self, encoder, decoder, input_text_processor, output_text_processor):

        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

        self.output_token_string_from_index = (
            tf.keras.layers.StringLookup(
                vocabulary=output_text_processor.get_vocabulary(),
                mask_token='',
                invert=True
            )
        )

        index_from_string = tf.keras.layers.StringLookup(
                            vocabulary=output_text_processor.get_vocabulary(), mask_token='')
        
        token_mask_ids = index_from_string(['','[UNK', '[START]']).numpy()
        token_mask = np.zeros([index_from_string.vocabulary_size()], dtype=np.bool)
        token_mask[np.array(token_mask_ids)] = True
        self.token_mask = token_mask

        self.start_token = index_from_string(tf.constant('[START]'))
        self.end_token = index_from_string(tf.constant('[END]'))

    def tokens_to_text(self, result_tokens):        
        
        result_text_tokens = self.output_token_string_from_index(result_tokens)        
        result_text = tf.strings.reduce_join(result_text_tokens, axis=1, separator=' ')        
        result_text = tf.strings.strip(result_text)        
        return result_text

    def sample(self, logits):
        
        #Sample from categorical distribution
        logits = tf.where(self.token_mask, -np.inf, logits)        
        logits = tf.squeeze(logits, axis=1)
        tokens = tf.random.categorical(logits, num_samples=1)        

        return tokens

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def tf_translate(self, input_text):
        return self.translate(input_text)
    
    def translate(self, input_text, *, max_length=50, return_attention=True):
        
        #Get encoder input and output
        batch_size = tf.shape(input_text)[0]
        input_tokens = self.input_text_processor(input_text)
        enc_output, enc_state = self.encoder(input_tokens)

        #Set decoder state to encoder output state
        dec_state = enc_state
        new_tokens = tf.fill([batch_size, 1], self.start_token)

        result_tokens = []
        attention = []
        done = tf.zeros([batch_size, 1], dtype=tf.bool)

        for itr in range(max_length):

            dec_input = DecoderInput(new_tokens=new_tokens, encoder_output=enc_output, mask=(input_tokens != 0))
            dec_result, dec_state = self.decoder(dec_input, state=dec_state)
            attention.append(dec_result.attention_weights)
            new_tokens = self.sample(dec_result.logits)

            #IF sequence produces end token, set to done
            done = done | (new_tokens == self.end_token)
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

            #Add output to result
            result_tokens.append(new_tokens)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        #Convert generated token ids to a list of strings
        result_tokens = tf.concat(result_tokens, axis=1)
        result_text = self.tokens_to_text(result_tokens)

        if return_attention:
            attention_stack = tf.concat(attention, axis=1)
            return {'text': result_text, 'attention': attention_stack}
        else:
            return {'text': result_text}
