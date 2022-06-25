from sacrebleu.metrics import BLEU, CHRF, TER
import tensorflow as tf
import tensorflow_text as tf_text
bleu = BLEU()

refs = [ # First set of references
       ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.']
             # Second set of references
       #['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.'],
          ]
#refs = ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.']

sys = ['The dog bit the man .', "It wasn't surprising .", 'The man had just bitten him .']

three_input_text = tf.constant([
    # Orlando Bloom and Miranda Kerr still love each other
    'Orlando Bloom und Miranda Kerr lieben sich noch immer',
    # Maria is very beautiful
    'Maria ist wunderschön.',    
    #My car is broken down	
    'Mein Auto ist kaputt.'
])

score = bleu.corpus_score(sys, refs)
print(score)