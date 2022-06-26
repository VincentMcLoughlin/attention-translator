import tensorflow as tf

class MaskedLoss(tf.keras.losses.Loss):

    def __init__(self):
        self.name = 'masked_loss'
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def __call__(self, y_true, y_pred):        

        #Batch loss
        loss = self.loss(y_true, y_pred)        

        #Mask losses on padding where our values are 0
        mask = tf.cast(y_true != 0, tf.float32)        
        loss *= mask

        return tf.reduce_sum(loss)