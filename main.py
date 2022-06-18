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
from attention import Attention
from Encoder import Encoder
from Decoder import Decoder, DecoderInput, DecoderOutput
import pickle 
from TrainTranslator import TrainTranslator
from MaskedLoss import MaskedLoss
from BatchLogs import BatchLogs
import time

new_data = True
embedding_dim = 256
units = 1024

def create_text_processors(dataset):        
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

print("getting example_input_batch")
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

attention_layer = Attention(units)
print((example_tokens != 0).shape)

example_attention_query = tf.random.normal(shape=[len(example_tokens), 2, 10])

context_vector, attention_weights = attention_layer(
    query=example_attention_query,
    value=example_enc_output,
    mask=(example_tokens != 0))

print(f'Attention result shape: (batch_size, query_seq_length, units):           {context_vector.shape}')
print(f'Attention weights shape: (batch_size, query_seq_length, value_seq_length): {attention_weights.shape}')

# plt.subplot(1, 2, 1)
# plt.pcolormesh(attention_weights[:, 0, :])
# plt.title('Attention weights')

# plt.subplot(1, 2, 2)
# plt.pcolormesh(example_tokens != 0)
# plt.title('Mask')

print(attention_weights.shape)

# attention_slice = attention_weights[0, 0].numpy()
# attention_slice = attention_slice[attention_slice != 0]

# plt.suptitle('Attention weights for one sequence')

# plt.figure(figsize=(12, 6))
# a1 = plt.subplot(1, 2, 1)
# plt.bar(range(len(attention_slice)), attention_slice)
# # freeze the xlim
# plt.xlim(plt.xlim())
# plt.xlabel('Attention weights')

# a2 = plt.subplot(1, 2, 2)
# plt.bar(range(len(attention_slice)), attention_slice)
# plt.xlabel('Attention weights, zoomed')

# # zoom in
# top = max(a1.get_ylim())
# zoom = 0.85*top
# a2.set_ylim([0.90*top, top])
# a1.plot(a1.get_xlim(), [zoom, zoom], color='k')
# plt.show()
# from Decoder import call
# Decoder.call = call
decoder = Decoder(output_text_processor.vocabulary_size(),
                  embedding_dim, units)

# Convert the target sequence, and collect the "[START]" tokens
example_output_tokens = output_text_processor(example_target_batch)

start_index = output_text_processor.get_vocabulary().index('[START]')
first_token = tf.constant([[start_index]] * example_output_tokens.shape[0])

print(first_token.shape)
print(example_enc_output.shape)
print((example_tokens != 0).shape)
print(example_enc_state.shape)
# Run the decoder
dec_result, dec_state = decoder(
    inputs = DecoderInput(new_tokens=first_token,
                          enc_output=example_enc_output,
                          mask=(example_tokens != 0)),
    state = example_enc_state
)

print(f'logits shape: (batch_size, t, output_vocab_size) {dec_result.logits.shape}')
print(f'state shape: (batch_size, dec_units) {dec_state.shape}')

sampled_token = tf.random.categorical(dec_result.logits[:, 0, :], num_samples=1)

vocab = np.array(output_text_processor.get_vocabulary())
first_word = vocab[sampled_token.numpy()]
first_word[:5]

dec_result, dec_state = decoder(
    DecoderInput(sampled_token,
                 example_enc_output,
                 mask=(example_tokens != 0)),
    state=dec_state)

sampled_token = tf.random.categorical(dec_result.logits[:, 0, :], num_samples=1)
first_word = vocab[sampled_token.numpy()]
first_word[:5]

translator = TrainTranslator(embedding_dim, units, 
                    input_text_processor=input_text_processor,
                    output_text_processor=output_text_processor,
                    use_tf_function=False)

translator.compile(optimizer = tf.optimizers.Adam(), loss=MaskedLoss())
np.log(output_text_processor.vocabulary_size())

#train without tf function
# start_time = time.time()
# for n in range(10):
#   print(translator.train_step([example_input_batch, example_target_batch]))
# print()
# end_time = time.time()
# elapsed = end_time - start_time
# print(f"Elapsed Time: {elapsed}")

# translator.use_tf_function = True
# #train with tf function
# start_time = time.time()
# for n in range(10):
#   print(translator.train_step([example_input_batch, example_target_batch]))
# print()
# end_time = time.time()
# elapsed = end_time - start_time
# print(f"Elapsed Time: {elapsed}")

# losses = []
# start_time = time.time()
# for n in range(100):
#   print('.', end='')
#   logs = translator.train_step([example_input_batch, example_target_batch])
#   losses.append(logs['batch_loss'].numpy())

# print()
# end_time = time.time()
# elapsed = end_time - start_time
# print(f"Elapsed Time: {elapsed}")
# plt.plot(losses)
# plt.show()

batch_loss = BatchLogs('batch_loss')

train_translator.fit(dataset, epochs=3, callbacks=[batch_loss])

plt.plot(batch_loss.logs)
plt.ylim([0, 3])
plt.xlabel('Batch #')
plt.ylabel('CE/token')
plt.show()