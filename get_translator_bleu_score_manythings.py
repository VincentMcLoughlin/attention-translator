import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import csv
from sacrebleu.metrics import BLEU
import sys 

#newstest2014
#https://nlp.stanford.edu/projects/nmt/

def clean_line(text):
    return_text = text.strip()
    return_text = return_text.replace('##AT##-##AT##', '')
    return_text = return_text.replace('&quot', '')
    return_text = return_text.replace('&apos', '')
    return return_text
    
def get_input_data_og(file_path, index=0):

    eng_lines = []
    deu_lines = []
    with open(file_path, newline = '',  encoding='utf-8') as file:
        line_reader = csv.reader(file, delimiter='\t')        
        for line in line_reader:

            eng_text = clean_line(line[0])
            deu_text = clean_line(line[1])            
            eng_lines.append(eng_text)
            deu_lines.append(deu_text)

    return eng_lines, deu_lines

#newstest2014
def get_input_data(file_path):

    lines = []    
    with open(file_path, newline = '',  encoding='utf-8') as file:        
        line_reader = csv.reader(file)
        for line in line_reader:                        
            text = line[0].strip()
            text = clean_line(line[0])         
            lines.append(text)

    return lines

use_attention = int(sys.argv[1])
if use_attention==0:  
    translator_path = 'train_split_translators\deu_eng_50k_1k_1layers_10_epochs_no_attention'
else:
    translator_path = 'train_split_translators\deu_eng_50k_1k_1layers_10_epochs'
  

reference_output_text, input_text = get_input_data_og('deu-eng/deu_val.txt')

reloaded = tf.saved_model.load(translator_path)

print("Loaded Model, translating input text")
print(f"input")
print(input_text)

print(f"Output")
reference_output_text 
print(reference_output_text)

start_index = 0
increment = 500 #About max for memory
end_index = start_index+increment
sys_output_text = []
while increment < len(input_text):
    batch_output_text = reloaded.tf_translate(tf.constant(input_text[start_index: end_index]))['text']
    batch_output_text = [x.numpy().decode() for x in batch_output_text]
    sys_output_text += batch_output_text

    start_index = end_index
    if end_index == len(input_text):
        break
    elif end_index + increment > len(input_text):
        end_index = len(input_text) 
    else:        
        end_index = end_index + increment

print(f"Sys output is")
print(sys_output_text)
print(len(sys_output_text))

bleu = BLEU(lowercase=True, tokenize='13a')
print("Calculating BLEU Score")
score = bleu.corpus_score(sys_output_text, [reference_output_text])
print(score)