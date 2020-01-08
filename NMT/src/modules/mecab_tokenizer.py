'''
Tokenize 1 line 1 sentence text using MeCab.

Usage:
python mecab_tokenizer.py -src hoge.txt -out hogehoge.txt
'''
import argparse
import codecs
import re
import MeCab

# command-line arguments
parser = argparse.ArgumentParser(description='make_parallel.py')
parser.add_argument('-src', required=True,
                    help='Path to the raw text data')
parser.add_argument('-out', required=True,
                    help='Output file name')
opt = parser.parse_args()


def main():
    m = MeCab.Tagger('-Owakati')
    srcF = codecs.open(opt.src, 'r', 'utf-8')
    outF = codecs.open(opt.out, 'w', 'utf-8')
    while True:
        sline = srcF.readline()
        if sline == '':
            break
        words = m.parse(sline)
        words = words.rstrip('\n')
        outF.write(words.strip()+'\n')
    srcF.close()
    outF.close()


if __name__ == '__main__':
    main()
