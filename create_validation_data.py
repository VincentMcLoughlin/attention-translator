
#from test_translator import clean_line, get_input_data_og
import random
import csv
import numpy as np

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

    return np.asarray(eng_lines), np.asarray(deu_lines)

input_file = 'deu-eng/deu_full.txt'
eng_lines, deu_lines = get_input_data_og(input_file)
total_lines = int(len(eng_lines))
print(total_lines)
num_val = int(total_lines*0.01)
print(f"Taking {num_val} samples")
val_indices = random.sample(range(0, total_lines), num_val)
print(len(val_indices))

full_data = np.vstack((eng_lines, deu_lines)).T
print(full_data.shape)
eng_val = eng_lines[val_indices]
deu_val = deu_lines[val_indices]
eng_train = np.delete(eng_lines, val_indices)
deu_train = np.delete(deu_lines, val_indices)
train_data = np.vstack((eng_train, deu_train)).T
val_data = np.vstack((eng_val, deu_val)).T

print(train_data.shape)
print(val_data.shape)

val_filename = "C:/Users/vince\Documents/Mechatronics/AI_Masters/EE8209 Intelligent Systems/Project/deu-eng/deu_val.txt"
np.savetxt(val_filename, val_data, delimiter="\t", fmt='%s', encoding='utf-8')
train_filename = "C:/Users/vince\Documents/Mechatronics/AI_Masters/EE8209 Intelligent Systems/Project/deu-eng/deu_train.txt"
np.savetxt(train_filename, train_data, delimiter="\t", fmt='%s', encoding='utf-8')