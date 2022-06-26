import numpy as np
from typing import Any, Tuple
import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from vectorize_text import clean_tf_text

def plot_attention(attention, sentence, predicted_sentence):
  sentence = clean_tf_text(sentence).numpy().decode().split()
  predicted_sentence = predicted_sentence.numpy().decode().split() + ['[END]']
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)

  attention = attention[:len(predicted_sentence), :len(sentence)]

  ax.matshow(attention, cmap='viridis', vmin=0.0)

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  ax.set_xlabel('Input text')
  ax.set_ylabel('Output text')
  plt.suptitle('Attention weights')
  plt.show()

user_input = str(sys.argv[2])
use_attention = int(sys.argv[1])

if use_attention==0:
  reloaded = tf.saved_model.load('train_split_translators\deu_eng_50k_1k_1layers_10_epochs')
else:
  reloaded = tf.saved_model.load('train_split_translators\deu_eng_50k_1k_1layers_10_epochs')

if user_input[-2:] != " ." and user_input[-1:] != ".":
  user_input += " ."
elif user_input[-1:] == ".":
  user_input = user_input.replace(".", " .")

print(f"USER {user_input}")
input_text = tf.constant([user_input])
print(f"INPUT: \n{input_text[0]}\n")
result = reloaded.tf_translate(input_text)

for tr in result['text']:
  print(f"OUTPUT: \n {tr.numpy().decode()}\n")

i=0
if use_attention:
  plot_attention(result['attention'][i], input_text[i], result['text'][i])