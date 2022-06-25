import numpy as np

import typing
from typing import Any, Tuple

import tensorflow as tf

import tensorflow_text as tf_text

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pathlib

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

def vectorize_text(input, max_vocab_size):
    
    text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punctuation, max_tokens=max_vocab_size)

    text_processor.adapt(input)

    return text_processor

## Text preprocessing

#Need to standardize 
# example_text = tf.constant('¿Todavía está en casa?')

# print(example_text.numpy())
# print(tf_text.normalize_utf8(example_text, 'NFKD').numpy())

# print(example_text.numpy().decode())
# print(tf_lower_and_split_punctuation(example_text).numpy().decode())