import numpy as np
import typing
from typing import Any, Tuple
import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pathlib
from create_dataset import Dataset
from vectorize_text import vectorize_text
from model_utils import save_processor, load_processor
from Encoder import Encoder
import pickle 

new_data = True
embedding_dim = 256
units = 1024

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# physical_devices = tf.config.experimental.list_physical_devices('GPU')

# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

def create_text_processors(dataset):
    print("start")
    example_input_batch = []
    for example_input_batch, example_target_batch in dataset.tf_dataset.take(1):
        print(example_input_batch[:5])
        print()
        print(example_target_batch[:5])
        example_input_batch = example_input_batch
        break

    print("getting input")
    input_path = "input_text_processor.pkl"

    if new_data:
        input_text_processor = vectorize_text(dataset.input)    
        save_processor(input_text_processor, input_path)
        print(input_text_processor.get_vocabulary()[:10])
    else:
        input_text_processor = load_processor(input_path, dataset.input)
        print(input_text_processor.get_vocabulary()[:10])

    print ("*"*10)

    print("getting target")
    output_path = "output_text_processor.pkl"

    if new_data:
        output_text_processor = vectorize_text(dataset.target)    
        save_processor(output_text_processor, output_path)
        print(output_text_processor.get_vocabulary()[:10])
    else:
        output_text_processor = load_processor(output_path, dataset.target)
        print(output_text_processor.get_vocabulary()[:10])

    print ("*"*10)
    return input_text_processor, output_text_processor

dataset = Dataset()
dataset.create_dataset()
input_text_processor, output_text_processor = create_text_processors(dataset)

# print("getting example")
for example_input_batch, example_target_batch in dataset.tf_dataset.take(1):
  print(example_input_batch[:5])
  print()
  print(example_target_batch[:5])
  break
example_tokens = input_text_processor(example_input_batch)
print(example_tokens[:3, :10])
# Encode the input sequence.
encoder = Encoder(input_text_processor.vocabulary_size(),
                  embedding_dim, units)
example_enc_output, example_enc_state = encoder(example_tokens)

print(f'Input batch, shape (batch): {example_input_batch.shape}')
print(f'Input batch tokens, shape (batch, s): {example_tokens.shape}')
print(f'Encoder output, shape (batch, s, units): {example_enc_output.shape}')
print(f'Encoder state, shape (batch, units): {example_enc_state.shape}')

print("getting vocabulary")
input_vocab = np.array(input_text_processor.get_vocabulary())
tokens = input_vocab[example_tokens[0].numpy()]
' '.join(tokens)

print(tokens)

plt.subplot(1, 2, 1)
plt.pcolormesh(example_tokens)
plt.title('Token IDs')

plt.subplot(1, 2, 2)
plt.pcolormesh(example_tokens != 0)
plt.title('Mask')
plt.show()