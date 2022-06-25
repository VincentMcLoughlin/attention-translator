import numpy as np
import typing
from typing import Any, Tuple
import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def tf_lower_and_split_punctuation(text):
    #split accented characters
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)

    #Keep space, a to z and some punctuation
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')

    #Add spaces around punctuation
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')

    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

def plot_attention(attention, sentence, predicted_sentence):
  sentence = tf_lower_and_split_punctuation(sentence).numpy().decode().split()
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

three_input_text = tf.constant([
    # Participants must then prove their skill and strength by carrying a bucket of water 50 metres .
    'Geschicklichkeit und Kraft müssen die Teilnehmer beim Wassereimertragen über 50 Meter unter Beweis stellen .',
    # His arrival was greeted with chants of , &quot; The army and the people are one hand , &quot; from the hundreds gathered in the university &apos;s main lecture room .
    'Bei seiner Ankunft wurde er von Hunderten Anwesenden im größten Hörsaal der Universität mit Sprechchören von „ Die Armee und das Volk sind eine Hand “ begrüßt .',        
    #'Let me call Tom'
    'Lassen Sie mich Tom anrufen .'
])

reloaded = tf.saved_model.load('train_split_translators\deu_eng_50k_1k_1layers_10_epochs')

result = reloaded.tf_translate(three_input_text)

for tr in result['text']:
  print(tr.numpy().decode())

i=2
plot_attention(result['attention'][i], three_input_text[i], result['text'][i])
print()