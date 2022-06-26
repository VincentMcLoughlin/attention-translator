import numpy as np

import typing
from typing import Any, Tuple

import tensorflow as tf

import tensorflow_text as tf_text

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def clean_tf_text(text):
    
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)    
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')

    #Add spaces around punctuation
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

def vectorize_text(input, max_vocab_size):
    
    text_processor = tf.keras.layers.TextVectorization(
    standardize=clean_tf_text, max_tokens=max_vocab_size)

    text_processor.adapt(input)

    return text_processor