import csv
import pathlib
import tensorflow as tf
import tensorflow_text as tf_text
rus_file = 'deu-eng/deu.txt'
# file1 = open(rus_file, 'r', encoding='utf-8')
# Lines = file1.readlines()
# print(Lines)

lines_list = []
with open(rus_file, newline = '',  encoding='utf-8') as games:
    line_reader = csv.reader(games, delimiter='\t')
    for line in line_reader:
        rus_eng = [line[0].strip(), line[1].strip()]        
        lines_list.append(rus_eng)

with open('output.txt', 'w', encoding='utf-8', newline='') as output:
    writer = csv.writer(output, delimiter='\t')
    writer.writerows(lines_list)