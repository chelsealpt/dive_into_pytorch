import collections
import re
#读入文本
def read_time_machine():
    with open('D:\\STUDY\\research_oversea\\France_research\\NLP\\1.txt', 'r') as f:
        for line in f:
            print(line)
        lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
    return lines
#strip用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列，在这里是用来删除单词之间的空格和换行符
#lower将字符串中的所有大写字母都转化成小写﻿

#re.sub()函数是用来字符串替换的函数﻿
# '[^a-z]+' 注意这里的^是非的意思，就是说非a-z字符串﻿
# 上面句子的含义是：将字符串str中的非小写字母开头的字符串以空格代替

lines = read_time_machine()
print('# sentences %d' % len(lines)
      
 #分词     
def tokenize(sentences, token='word'):
    """Split sentences into word or char tokens"""
    if token == 'word': #对单词级别进行分词
        return [sentence.split(' ') for sentence in sentences]
    elif token == 'char': #对字符级别进行分词
        return [list(sentence) for sentence in sentences]
    else:
        print('ERROR: unkown token type '+token)

#建立字典
class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        #tokens是一个二维列表，min_freq表示阈值，出现小于几的可以忽略
        counter = count_corpus(tokens)  #key，value : 词，词频，已去重
        self.token_freqs = list(counter.items()) #返回一个list
        self.idx_to_token = []
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['<pad>', '<bos>', '<eos>', '<unk>']
            #pad:二维矩阵长度不一，短句子补token利用pad
            #bos:开始token
            #eos：结束token
            #unk：未登录词当作unk

        else:
            self.unk = 0
            self.idx_to_token += ['<unk>']
        self.idx_to_token += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in self.idx_to_token]#索引到词的映射
        #形成词到索引的映射               
        self.token_to_idx = dict()
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx
            

    def __len__(self):
        return len(self.idx_to_token)
     
     #词到索引的映射
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
   #给定索引，返回对应的词
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(sentences):
    tokens = [tk for st in sentences for tk in st]#讲二维展开成一维
    '''
    等同于
    tokens=[]
    for st in sentences:
    for tk in st:
        tokens.append(tk)
    '''
    return collections.Counter(tokens)  # 返回一个字典，记录每个词的出现次数
#实现
vocab = Vocab(tokens)#构建vocab实例
print(list(vocab.token_to_idx.items())[0:10]) 

#将词转换成索引
for i in range(8, 10):
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])

#用现有工具分词
text = "Mr. Chen doesn't agree with my suggestion."
#spacy
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
print([token.text for token in doc])
#NLTK
from nltk.tokenize import word_tokenize
from nltk import data
data.path.append('/home/kesci/input/nltk_data3784/nltk_data')
print(word_tokenize(text))
