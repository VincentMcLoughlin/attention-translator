import tensorflow as tf

class Attention(tf.keras.layers.Layer):

    def __init__(self, units):
        super().__init__()
        
        #Our W1 parameter in the score function equation
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)

        #Our W2 parameter in the score function equation
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        # tf implementation of Bahdanau's additive attention
        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):        

        w1_ht = self.W1(query) #query = h_t in our equation
        w2_hs = self.W2(value) #value = h_s in our equation

        ht_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)

        hs_mask = mask

        inputs = [w1_ht, value, w2_hs]
        masks = [ht_mask, hs_mask]

        context_vector, attention_weights = self.attention(inputs, masks, return_attention_scores=True)        

        return context_vector, attention_weights