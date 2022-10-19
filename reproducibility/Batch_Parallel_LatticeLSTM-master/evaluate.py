import os
from load_data import load_ontonotes4ner, equip_chinese_ner_with_skip, load_yangjie_rich_pretrain_word_list, \
    load_resume_ner, load_weibo_ner, load_weibo_ner_old, load_ppzh_ner, load_zhPrivacyPolicyner
from fastNLP.io import ModelLoader
from fastNLP import SpanFPreRecMetric, Trainer, AccuracyMetric, LossInForward, LRScheduler, Tester, \
    ClassifyFPreRecMetric, ConfusionMatrixMetric
from fastNLP.io.loader import ConllLoader
from fastNLP import Vocabulary
from fastNLP.embeddings import StaticEmbedding
from tqdm import tqdm

yangjie_rich_pretrain_bigram_path = '/home/zachary2/Documents/NER/Batch_Parallel_LatticeLSTM-master/pre_embeddings/gigaword_chn.all.a2b.bi.ite50.vec'
yangjie_rich_pretrain_unigram_path = '/home/zachary2/Documents/NER/Batch_Parallel_LatticeLSTM-master/pre_embeddings/gigaword_chn.all.a2b.uni.ite50.vec'
ppzh_ner_path = '/data_SSD/zachary/NLP/zhPrivacyPolicyNER'
from utils_ import LatticeLexiconPadder

yangjie_rich_pretrain_word_path = '/home/zachary2/Documents/NER/Batch_Parallel_LatticeLSTM-master/pre_embeddings/ctb.50d.vec'

import argparse

# parameter
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda')
parser.add_argument('--debug', default=False)
#
parser.add_argument('--norm_embed', default=False)
parser.add_argument('--batch', default=4)
parser.add_argument('--test_batch', default=512)
parser.add_argument('--optim', default='sgd', help='adam|sgd')
parser.add_argument('--lr', default=0.045)
parser.add_argument('--model', default='lattice', help='lattice|lstm')
parser.add_argument('--skip_before_head', default=False)  # in paper it's false
parser.add_argument('--hidden', default=113)
parser.add_argument('--momentum', default=0)
parser.add_argument('--bi', default=True)
parser.add_argument('--dataset', default='ppzh', help='resume|ontonote|weibo|msra|ppzh')
parser.add_argument('--use_bigram', default=True)
#
parser.add_argument('--embed_dropout', default=0.5)
parser.add_argument('--gaz_dropout', default=-1)
parser.add_argument('--output_dropout', default=0.5)
parser.add_argument('--epoch', default=100)
parser.add_argument('--seed', default=100)
# kaifa
parser.add_argument('--predict', default=True)
# kaifa 2022-06-02
parser.add_argument("--local_rank", type=int)

args = parser.parse_args()


def get_bigrams(words):
    result = []
    for i, w, in enumerate(words):
        if i != len(words) - 1:
            result.append(words[i] + words[i + 1])
        else:
            result.append(words[i] + '<end>')
    return result


loader = ConllLoader(['chars', 'target'])


def load_zhPrivacyPolicyner(path, char_embedding_path=None, bigram_embedding_path=None, index_token=True,
                            normalize={'char': True, 'bigram': True, 'word': False}):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    train_path = os.path.join(path, 'train.char.bmes')
    dev_path = os.path.join(path, 'dev.char.bmes')
    test_path = os.path.join(path, 'test.char.bmes')

    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None, unknown=None)
    print(label_vocab)
    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test']])
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    print(label_vocab)
    if index_token:
        char_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                 field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                   field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                  field_name='target', new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['label'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
                                         normalize=normalize['char'])
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
                                           normalize=normalize['bigram'])
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings


datasets, vocabs, embeddings = load_zhPrivacyPolicyner(ppzh_ner_path, yangjie_rich_pretrain_unigram_path,
                                                       yangjie_rich_pretrain_bigram_path,
                                                       index_token=False,
                                                       )
refresh_data = False
w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                              _refresh=refresh_data)

cache_name = os.path.join('cache', args.dataset + '_lattice')
datasets, vocabs, embeddings = equip_chinese_ner_with_skip(datasets, vocabs, embeddings, w_list,
                                                           yangjie_rich_pretrain_word_path,
                                                           _refresh=refresh_data, _cache_fp=cache_name)

for k, v in datasets.items():
    if args.model == 'lattice':
        v.set_ignore_type('skips_l2r_word', 'skips_l2r_source', 'skips_r2l_word', 'skips_r2l_source')
        if args.skip_before_head:
            v.set_padder('skips_l2r_word', LatticeLexiconPadder())
            v.set_padder('skips_l2r_source', LatticeLexiconPadder())
            v.set_padder('skips_r2l_word', LatticeLexiconPadder())
            v.set_padder('skips_r2l_source', LatticeLexiconPadder(pad_val_dynamic=True))
        else:
            v.set_padder('skips_l2r_word', LatticeLexiconPadder())
            v.set_padder('skips_r2l_word', LatticeLexiconPadder())
            v.set_padder('skips_l2r_source', LatticeLexiconPadder(-1))
            v.set_padder('skips_r2l_source', LatticeLexiconPadder(pad_val_dynamic=True, dynamic_offset=1))
        if args.bi:
            v.set_input('chars', 'bigrams', 'seq_len',
                        'skips_l2r_word', 'skips_l2r_source', 'lexicon_count',
                        'skips_r2l_word', 'skips_r2l_source', 'lexicon_count_back',
                        'target',
                        use_1st_ins_infer_dim_type=True)
        else:
            v.set_input('chars', 'bigrams', 'seq_len',
                        'skips_l2r_word', 'skips_l2r_source', 'lexicon_count',
                        'target',
                        use_1st_ins_infer_dim_type=True)
        v.set_target('target', 'seq_len')

        v['target'].set_pad_val(0)
    elif args.model == 'lstm':
        v.set_ignore_type('skips_l2r_word', 'skips_l2r_source')
        v.set_padder('skips_l2r_word', LatticeLexiconPadder())
        v.set_padder('skips_l2r_source', LatticeLexiconPadder())
        v.set_input('chars', 'bigrams', 'seq_len', 'target',
                    use_1st_ins_infer_dim_type=True)
        v.set_target('target', 'seq_len')

        v['target'].set_pad_val(0)
import torch

model1 = ModelLoader.load_pytorch_model("./cache/model_ckpt_100_2.pkl")
tester1 = Tester(datasets['test'], model1, metrics=ConfusionMatrixMetric(), device='cuda', batch_size=512)
a = tester1.test()

tester2 = Tester(datasets['test'], model1, metrics=ClassifyFPreRecMetric(only_gross=False), device='cuda', batch_size=512)
tester2.test()




# model2 = torch.load("./cache/model_ckpt_100_2.pkl")
# data_iterator = tester1.data_iterator
#
# eval_results = []
#
# device='cuda'
# for x in datasets['test']:
#     x = x.to(device)
#     y = y.to(device)
#     pred = model1(x)
