import os
import random as rd
from shutil import copyfile

DIR_1 = "input"
DIR_2 = "test_words"
DIR_OUTPUT = "mix"
BLOCK_SIZE = 10
WORDS = os.listdir(DIR_2)

paths_1 = list(filter(lambda p: p[0] != "_" ,os.listdir(DIR_1)))
paths_2 = os.listdir(DIR_2) 

for i in range(len(WORDS)):
    try:
        os.mkdir(os.path.join(DIR_OUTPUT,WORDS[i]))
    except Exception:
        pass
    files_1 = os.listdir(os.path.join(DIR_1,paths_1[i]))
    files_2 = os.listdir(os.path.join(DIR_2,paths_2[i]))
    index_1 = rd.randint(0,len(files_1) - 3*BLOCK_SIZE)
    index_2 = rd.randint(0,len(files_2) - BLOCK_SIZE)
    for file in files_1[index_1: index_1 + 3*BLOCK_SIZE]:
        from_ = os.path.join(DIR_1,WORDS[i],file)
        to_ = os.path.join(DIR_OUTPUT,WORDS[i],file)
        copyfile(from_,to_)
    for file in files_2[index_2:index_2 + BLOCK_SIZE]:
        from_ = os.path.join(DIR_2,WORDS[i],file)
        to_ = os.path.join(DIR_OUTPUT,WORDS[i],file)
        copyfile(from_,to_)

    
    