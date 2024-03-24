#----------DEPENDENCIES----------#
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path


#----------FUNCTION TO BUILD TOKENIZER---------------#
def get_all_sentences(data, lang):
    for idx in range(len(data)):
        yield data[idx]['translation'][lang]#-->modify the logic to get the sentences from your dataset

def get_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) 
    if not Path.exists(tokenizer_path):     #-->creates the tokenizer if not already present
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))  #-->loads the tokenizer if already available
    return tokenizer

