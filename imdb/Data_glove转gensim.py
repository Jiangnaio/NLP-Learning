"""
将glove模型转换成gensim方便加载的格式(gensim支持word2vec格式的预训练模型格式）
"""

# def glove_to_gensim(glove_file, gensim_file):
#     with open(glove_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#     with open(gensim_file, 'w', encoding='utf-8') as f:
#         for line in lines:
#             word, vector = line.split(' ', 1)
#             vector = vector.strip()
#             f.write(f"{word} {vector}\n")
# if __name__ == '__main__':
#     glove_file = 'data/glove.840B.300d.txt'
#     gensim_file = 'data/glove.840B.300d.gensim.txt' 
#     glove_to_gensim(glove_file, gensim_file)

import os
# 用于转换并加载glove预训练词向量
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
# 将glove转换为word2vec
from gensim.scripts.glove2word2vec import glove2word2vec
path=os.getcwd()
glove_file=datapath(os.path.join(path, "data/glove.840B.300d.txt"))
tmp_file=get_tmpfile(os.path.join(path,"data/word2vec.txt"))
glove2word2vec(glove_file, tmp_file)