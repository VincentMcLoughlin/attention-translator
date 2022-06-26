import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
from create_dataset import Dataset
from vectorize_text import vectorize_text
from TrainTranslator import TrainTranslator
from MaskedLoss import MaskedLoss
from BatchLogs import BatchLogs
from Translator import Translator

embedding_dim = 1000
units = 1000
max_vocab_size = 50000
batch_size = 128
learning_rate = 1e-3
num_epochs = 10

def create_text_processors(dataset):        
    print("getting input")    

    input_text_processor = vectorize_text(dataset.input, max_vocab_size)            

    print ("*"*10)

    print("getting target")
    output_text_processor = vectorize_text(dataset.target, max_vocab_size)            

    print ("*"*10)
    return input_text_processor, output_text_processor


def main():

    #Get Data
    german_path = 'deu-eng/deu_train.txt'
    german_val_path = 'deu-eng/deu_val.txt'
    dataset = Dataset()
    training_data = dataset.create_dataset(german_path)
    #val_data, val_inp_data, val_targ_data = dataset.create_dataset(german_val_path)

    #Create Text Processors
    input_text_processor, output_text_processor = create_text_processors(dataset)

    #Train translator
    train_translator = TrainTranslator(embedding_dim, units, 
                        input_text_processor=input_text_processor,
                        output_text_processor=output_text_processor)

    train_translator.compile(optimizer = tf.optimizers.Adam(learning_rate=learning_rate), loss=MaskedLoss())    
    batch_loss = BatchLogs('batch_loss')
    train_translator.fit(training_data, epochs=num_epochs, callbacks=[batch_loss], batch_size=batch_size)
    
    #Create translator model
    translator = Translator(
        encoder=train_translator.encoder,
        decoder=train_translator.decoder,
        input_text_processor=input_text_processor,
        output_text_processor=output_text_processor,
    )

    tf.saved_model.save(translator, 'translator',
                        signatures={'serving_default': translator.tf_translate})

    #Plot loss
    plt.plot(batch_loss.logs)
    plt.ylim([0, 3])
    plt.xlabel('Batch #')
    plt.ylabel('Cross Entropy Loss')
    plt.show()
    

if __name__ == "__main__":
    main()