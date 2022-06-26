# attention-translator

Translator that translates from German to English using TensorFlow's additive attention, following the description in Luong et al.'s 2015 paper (https://arxiv.org/pdf/1508.04025v5.pdf)

## Demo

To demo the code, first you will need to activate the virtual environment. This can be done by using 

```
source venv/Scripts/activate
```

in git bash (if you are using windows) or linux systems. Equivalent powershell commands should also work.


To demo the code, use python German_to_English_Translate_and_Plot.py. This will use our attention translator and plot the attention weights for the source and target sentence.

NOTE: Performance is better when the sentence is formatted as shown in the example, with a space before the period.

For example 

```
python German_to_English_Translate_and_Plot.py "Sie können mein Haus von hier nicht sehen ."
```

Example sentences can be found in our validation data set, which is located in deu-eng/deu_val.txt. The newstest2014 validation set was also used to validate our network, and can be found in the test_data/newstest2014_deu.txt and test_data/newstest2014_eng.txt. See the following web pages for the source of the newstest2014 dataset, which also contains more datasets if required.

https://nlp.stanford.edu/projects/nmt/

https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de

https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en

Our translation models can be found in the train_split_translators folder and are labeled clearly.

## Run Training
To perform a training run, simply call
```
python main.py
```

The code is currently set up to use the same training parameters mentioned in the paper. Specifically, it is a 10 epoch run, with 1 1000 node GRU layer in both the encoder and decoder. It uses a vocabulary size of 50,000 and an embedding dimension size of 1000 with a batch size of 128. All of these parameters can be changed from the top of main.py if so desired.

It takes about 10 minutes on my RTX 3070 GPU to create the input and output text processors and after that each training run is about 25 minutes.
