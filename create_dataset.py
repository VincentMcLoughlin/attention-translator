import numpy as np

import typing
from typing import Any, Tuple

import tensorflow as tf

import tensorflow_text as tf_text

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pathlib

class Dataset():

    def __init__(self):
        self.input = []
        self.target = []
        self.tf_dataset = []

    def load_data(self, path):
        text = path.read_text(encoding='utf-8')
        lines = text.splitlines()

        pairs = [line.split('\t') for line in lines]

        inp = [inp for targ, inp in pairs]
        targ = [targ for targ, inp in pairs]        

        return inp, targ

    def create_dataset(self, file_path, download=False):

        if download == True:
            path_to_zip = tf.keras.utils.get_file('spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip', extract=True)            
            path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng/spa.txt'

        path_to_file = pathlib.Path(file_path)

        self.input, self.target = self.load_data(path_to_file)
        print(self.input[1])
        print(self.target[-1])

        BUFFER_SIZE = len(self.input)
        BATCH_SIZE = 64

        dataset = tf.data.Dataset.from_tensor_slices((self.input, self.target)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE)
        self.tf_dataset = dataset
        tf.data.experimental.save(
            self.tf_dataset, "spanish_english_dataset", compression=None, shard_func=None, checkpoint_args=None
        )

        print(f"Saved dataset")
        return dataset, self.input, self.target