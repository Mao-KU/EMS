'''
data cleaning
1. remove sentences covering other characters in raw corpus
2. train bpe tokenizer and store model and vocab.
'''
from tqdm import tqdm
import os
import sentencepiece as spm
import logging
import sys
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


raw_path = sys.argv[1] # set the path of raw dataset here

french_letters = ['é',
        'à', 'è', 'ù',
        'â', 'ê', 'î', 'ô', 'û',
        'ç',
        'ë', 'ï', 'ü']

german_letters = ['ä', 'ö', 'ü',  'ß']

italian_letters = ['à', 'è', 'ì', 'ò', 'ù', 'é', 'ó', 'î']

spanish_letters = ['á', 'é', 'í', 'ó', 'ú', 'ñ', 'ü', '¿', '¡', 'ç', '·', 'º', 'ª']

commas = ['.','?','!',',','"',':',';','<','>','-','+','=',' ','_', "'",'%','&','#','@','*','(',')','~','`']
nums = list('0123456789')
allowed = list('abcdefghijklmnopqrstuvwxyz') + french_letters + commas + nums


def remove_other_letters(text):
    if '_en' == text[-3:] or '_fr' == text[-3:]:
        if text[0] in allowed and text[-3] in allowed:
            return text[:-3]
    elif '<<split>>' == text:
        return '<<split>>'
    return ''

def write_cleaned_data(raw_path):
    file_name = '/loquat/mao/ParaCrawl/'+ raw_path.split('/')[-1] + '_bpe'
    f_processed = open(file_name, 'w+')
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Cleaning raw corpus"):
            words = line.split(' ')
            processed = list(map(lambda word: remove_other_letters(word), words))
            index = processed.index('<<split>>')
            f_processed.write(' '.join(processed[:index])+'\n' + ' '.join(processed[index+1:]))  
    f_processed.close()
    logger.info('Finished writing corpus line by line in {}.'.format(file_name))
    return file_name

def train_bpe(bpe_path):
    spm.SentencePieceTrainer.Train('--input={} \
                                    --model_prefix=62languages_bpe_60000 \
                                    --vocab_size=60000 \
                                    --model_type=bpe \
                                    --pad_id=0 \
                                    --unk_id=1 \
                                    --bos_id=-1 \
                                    --eos_id=-1 \
                                    --control_symbols=<MASK>,<2en>,<2af>,<2ar>,<2bg>,<2bn>,<2ca>,<2cs>,<2da>,<2de>,<2el>,<2eo>,<2es>,<2et>,<2eu>,<2fa>,<2fi>,<2fr>,<2gl>,<2gu>,<2he>,<2hi>,<2hr>,<2hu>,<2hy>,<2id>,<2it>,<2ja>,<2jv>,<2ka>,<2kk>,<2ko>,<2ku>,<2lt>,<2lv>,<2ml>,<2mr>,<2ms>,<2mk>,<2mn>,<2my>,<2nl>,<2nb>,<2pl>,<2pt>,<2ro>,<2ru>,<2sk>,<2sl>,<2sq>,<2sr>,<2sv>,<2sw>,<2ta>,<2te>,<2th>,<2tl>,<2tr>,<2uk>,<2ur>,<2vi>,<2yo>,<2zh>'
                                   .format(bpe_path))

if __name__ == "__main__":
    # path = write_cleaned_data(raw_path)
    path = raw_path
    train_bpe(path)
    logger.info('Finished training bpe tokenizer.')

