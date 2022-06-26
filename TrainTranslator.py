import tensorflow as tf
from Encoder import Encoder
from Decoder import Decoder, DecoderInput

class TrainTranslator(tf.keras.Model):

    def __init__(self, embedding_dim, units, input_text_processor, output_text_processor):
        super().__init__()

        encoder = Encoder(units, input_text_processor.vocabulary_size(), embedding_dim)
        decoder = Decoder(units, output_text_processor.vocabulary_size(), embedding_dim)

        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor        

    def _train_step(self, inputs):
        input_text, target_text = inputs

        (input_tokens, input_mask, target_tokens, target_mask) = self._preprocess(input_text, target_text)

        target_max_length = tf.shape(target_tokens)[1]

        with tf.GradientTape() as tape:

            #encode input, enc_state is passed to decoder
            enc_output, enc_state = self.encoder(input_tokens)            

            #Initialize decoder state to encoder's final state            
            dec_state = enc_state
            loss = tf.constant(0.0)

            for t in tf.range(target_max_length-1):            
                #Pass in two tokens from target sequence
                # 1. The current input to the decoder
                # 2. The target for the decoder's next prediction
                new_tokens = target_tokens[:, t:t+2]                
                step_loss, dec_state = self._loop_step(new_tokens, input_mask, enc_output, dec_state)
                loss = loss+step_loss

            #Get average loss from non-zero tokens
            avg_loss = loss/tf.reduce_sum(tf.cast(target_mask, tf.float32))            

        #Apply optimization
        variables = self.trainable_variables
        gradients = tape.gradient(avg_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return {'batch_loss': avg_loss}

    def train_step(self, inputs):        

        return self._tf_train_step(inputs)        

    def _preprocess(self, input_text, target_text):        

        #Convert the text to tokens
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)        
        #Convert IDs to masks
        input_mask = input_tokens != 0        
        target_mask = target_tokens != 0        

        return input_tokens, input_mask, target_tokens, target_mask

    def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
        input_token, target_token = new_tokens[:,0:1], new_tokens[:, 1:2]

        #Run the decoder one step
        decoder_input = DecoderInput(new_tokens=input_token, 
                                    encoder_output=enc_output, 
                                    mask=input_mask)

        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)        

        y = target_token
        y_pred = dec_result.logits
        step_loss = self.loss(y, y_pred)

        return step_loss, dec_state

    @tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]),
                               tf.TensorSpec(dtype=tf.string, shape=[None])]])
    def _tf_train_step(self, inputs):
        return self._train_step(inputs)
        