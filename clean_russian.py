import csv
import pathlib
import tensorflow as tf
import tensorflow_text as tf_text
rus_file = 'rus-eng/rus.txt'
# file1 = open(rus_file, 'r', encoding='utf-8')
# Lines = file1.readlines()
# print(Lines)

lines_list = []
with open(rus_file, newline = '',  encoding='utf-8') as games:
    line_reader = csv.reader(games, delimiter='\t')
    for line in line_reader:
        rus_eng = [line[0].strip(), line[1].strip()]        
        lines_list.append(rus_eng)

# with open('output.txt', 'w', encoding='utf-8', newline='') as output:
#     writer = csv.writer(output, delimiter='\t')
#     writer.writerows(lines_list)

file_path = rus_file
path = pathlib.Path(file_path)
text = path.read_text(encoding='utf-8')
lines = text.splitlines()

pairs = [line.split('\t') for line in lines]

inp = [inp for targ, inp in pairs]
targ = [targ for targ, inp in pairs]        

def decode_string(ints):
  strs = [chr(i) for i in ints]
  joined = [''.join(strs)]
  return joined

text = [inp[100]]
print(text)

text = tf_text.normalize_utf8(text, 'NFKD')
text = tf.strings.regex_replace(text, '\S*@\S*\s?\S*#', '')
text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
text = tf.strings.strip(text)
text = tf.strings.unicode_decode(text, 'utf-8')
decoded_list = [decode_string(ex) for ex in text]
text = tf.cast(decoded_list, tf.string)
print(text)

#text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
#text = tf.strings.regex_replace(text, '\S*@\S*\s?\S*#', '')
#str = re.sub('\S*@\S*\s?\S*#', '', str) #Remove emails/special characters
#print(decoded_list)

text = targ[100]
print(text)
text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')        
#Add spaces around punctuation
text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
text = tf.strings.strip(text)
print(text)

#return inp, targ