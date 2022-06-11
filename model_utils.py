import pickle
import tensorflow_text as tf_text
import tensorflow as tf

def load_processor(model_path, input):

    model_data = pickle.load(open(model_path, "rb"))
    text_processor = tf.keras.layers.TextVectorization.from_config(model_data['config'])
    text_processor.adapt(input)
    return text_processor
    
def save_processor(text_processor, path):
    pickle.dump({'config': text_processor.get_config(),
             'weights': text_processor.get_weights()}
            , open(path, "wb"))