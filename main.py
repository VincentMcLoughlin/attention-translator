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
from Translator import Translator
import time

new_data = True
embedding_dim = 1000
units = 1000
max_vocab_size = 50000
batch_size = 128
learning_rate = 1e-3
num_epochs = 4
#learning_rate = 0.01
def create_text_processors(dataset, use_russian=False):        
    print("getting input")
    input_path = "input_text_processor.pkl"

    if new_data:
        input_text_processor = vectorize_text(dataset.input, max_vocab_size)    
        save_processor(input_text_processor, input_path)
        print(input_text_processor.get_vocabulary()[:10])
    else:
        input_text_processor = load_processor(input_path, dataset.input)
        print(input_text_processor.get_vocabulary()[:10])

    print ("*"*10)

    print("getting target")
    output_path = "output_text_processor.pkl"

    if new_data:
        output_text_processor = vectorize_text(dataset.target, max_vocab_size)    
        save_processor(output_text_processor, output_path)
        print(output_text_processor.get_vocabulary()[:10])
    else:
        output_text_processor = load_processor(output_path, dataset.target)
        print(output_text_processor.get_vocabulary()[:10])

    print ("*"*10)
    return input_text_processor, output_text_processor

spanish_path = 'spa-eng/spa.txt'
russian_path = 'rus-eng/rus.txt'
german_path = 'deu-eng/deu_train.txt'
german_val_path = 'deu-eng/deu_val.txt'
dataset = Dataset()
training_data, inp_data, targ_data = dataset.create_dataset(german_path)
val_data, val_inp_data, val_targ_data = dataset.create_dataset(german_val_path)
input_text_processor, output_text_processor = create_text_processors(dataset, use_russian=True)

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

# attention_layer = Attention(units)
# print((example_tokens != 0).shape)

# example_attention_query = tf.random.normal(shape=[len(example_tokens), 2, 10])

# context_vector, attention_weights = attention_layer(
#     query=example_attention_query,
#     value=example_enc_output,
#     mask=(example_tokens != 0))

# print(f'Attention result shape: (batch_size, query_seq_length, units):           {context_vector.shape}')
# print(f'Attention weights shape: (batch_size, query_seq_length, value_seq_length): {attention_weights.shape}')

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

train_translator = TrainTranslator(embedding_dim, units, 
                    input_text_processor=input_text_processor,
                    output_text_processor=output_text_processor, use_tf_function=True)

train_translator.compile(optimizer = tf.optimizers.Adam(learning_rate=learning_rate), loss=MaskedLoss())
np.log(output_text_processor.vocabulary_size())

#train without tf function
print(f"example_input_batch shape {example_input_batch.shape}")
print(f"target_batch_shape {example_target_batch.shape}")
#print(f"tf_dataset shape {dataset.tf_dataset.shape}")
#print(f"training data shape {training_data.shape}")
start_time = time.time()
for n in range(10):
  print(train_translator.train_step([example_input_batch, example_target_batch]))
print()
end_time = time.time()
elapsed = end_time - start_time
print(f"Elapsed Time: {elapsed}")

batch_loss = BatchLogs('batch_loss')
#train_translator.fit(training_data, validation_data = val_data, epochs=num_epochs, callbacks=[batch_loss], batch_size=batch_size)
train_translator.fit(training_data, epochs=num_epochs, callbacks=[batch_loss], batch_size=batch_size)
                   
translator = Translator(
    encoder=train_translator.encoder,
    decoder=train_translator.decoder,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor,
)

tf.saved_model.save(translator, 'translator',
                    signatures={'serving_default': translator.tf_translate})

plt.plot(batch_loss.logs)
plt.ylim([0, 3])
plt.xlabel('Batch #')
plt.ylabel('Cross Entropy Loss')
plt.show()

print(f"model history is \n {train_translator.history}")

print(f"translator history is \n {translator.history}")

print("TRAINER SUMMARY")
train_translator.summary()
print("TRANSLATOR SUMMARY")
translator.summary()