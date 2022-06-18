from tkinter import Variable
import tensorflow as tf
from Encoder import Encoder
from Decoder import Decoder, DecoderInput
from ShapeChecker import ShapeChecker

class TrainTranslator(tf.keras.Model):

    def __init__(self, embedding_dim, units, input_text_processor, 
                output_text_processor, use_tf_function=True):
        super().__init__()

        encoder = Encoder(input_text_processor.vocabulary_size(), embedding_dim, units)
        decoder = Decoder(output_text_processor.vocabulary_size(), embedding_dim, units)

        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.use_tf_function = use_tf_function
        self.shape_checker = ShapeChecker()

    def _train_step(self, inputs):
        input_text, target_text = inputs

        (input_tokens, input_mask, target_tokens, target_mask) = self._preprocess(input_text, target_text)

        max_target_length = tf.shape(target_tokens)[1]

        with tf.GradientTape() as tape:

            #encode input
            enc_output, enc_state = self.encoder(input_tokens)
            self.shape_checker(enc_output, ('batch', 's', 'enc_units'))
            self.shape_checker(enc_state, ('batch', 'enc_units'))

            #Initialize decoder state to encoder's final state
            #Only works if encoder and decoder have same number of units            
            dec_state = enc_state
            loss = tf.constant(0.0)

            for t in tf.range(max_target_length-1):
                #Pass in two tokens from target sequence
                # 1. The current input to the decoder
                # 2. The target for the decoder's next prediction
                new_tokens = target_tokens[:, t:t+2]                
                step_loss, dec_state = self._loop_step(new_tokens, input_mask, enc_output, dec_state)

                loss = loss+step_loss

            #Average loss over all non padding tokens
            average_loss = loss/tf.reduce_sum(tf.cast(target_mask, tf.float32))            

        #Apply  optimization step
        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        #Return dict mapping metric names to current value
        return {'batch_loss': average_loss}

    def train_step(self, inputs):        

        if self.use_tf_function:
            return self._tf_train_step(inputs)
        else:
            return self._train_step(inputs)

    def _preprocess(self, input_text, target_text):
        self.shape_checker(input_text, ('batch',))
        self.shape_checker(input_text, ('batch',))

        #Convert the text to token IDs
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)

        self.shape_checker(input_tokens, ('batch', 's'))
        self.shape_checker(target_tokens, ('batch','t'))

        #Convert IDs to masks
        input_mask = input_tokens != 0
        self.shape_checker(input_mask, ('batch', 's'))

        target_mask = target_tokens != 0
        self.shape_checker(target_mask, ('batch', 't'))

        return input_tokens, input_mask, target_tokens, target_mask

    def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
        input_token, target_token = new_tokens[:,0:1], new_tokens[:, 1:2]

        #Run the decoder one step
        decoder_input = DecoderInput(new_tokens=input_token, 
                                    enc_output=enc_output, 
                                    mask=input_mask)

        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)

        self.shape_checker(dec_result.logits, ('batch', 't1', 'logits'))
        self.shape_checker(dec_result.attention_weights, ('batch', 't1', 's'))
        self.shape_checker(dec_state, ('batch', 'dec_units'))

        y = target_token
        y_pred = dec_result.logits
        step_loss = self.loss(y, y_pred)

        return step_loss, dec_state

    @tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]),
                               tf.TensorSpec(dtype=tf.string, shape=[None])]])
    def _tf_train_step(self, inputs):
        return self._train_step(inputs)
        