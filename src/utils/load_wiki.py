import sys
import os
import random


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.utils.download_tool import DATA_HUB, download_extract, DATA_URL


# 使用完整明确的URL
DATA_HUB['wikitext-2'] = (
DATA_URL + 
'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')



def read_wiki(dir):
    file_name = os.path.join(dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def load_data_wiki():
    dir = download_extract('wikitext-2', 'wikitext-2')
    return read_wiki(dir)
    



if __name__ == "__main__":
    paragraphs = load_data_wiki()
    


