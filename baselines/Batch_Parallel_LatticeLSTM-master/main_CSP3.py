
import os
import argparse

from fastNLP.io import ModelSaver, ModelLoader

from utils import set_seed
from utils_ import LatticeLexiconPadder
from models import LSTM_SeqLabel, LatticeLSTM_SeqLabel_V1
from fastNLP import SpanFPreRecMetric, Trainer, AccuracyMetric, LossInForward, LRScheduler, Tester
from fastNLP.core.callback import FitlogCallback
import fitlog
import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch.optim as optim
from pathes import *
from load_data import load_ontonotes4ner, equip_chinese_ner_with_skip, load_yangjie_rich_pretrain_word_list, \
    load_resume_ner, load_weibo_ner, load_weibo_ner_old, load_ppzh_ner, load_zhPrivacyPolicyner


## kz init multi gpu torch



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
#
if __name__ == '__main__':
    set_seed(args.seed)

    #
    fit_msg_list = [args.model, 'bi' if args.bi else 'uni', str(args.batch)]
    if args.model == 'lattice':
        fit_msg_list.append(str(args.skip_before_head))
    fit_msg = ' '.join(fit_msg_list)
    fitlog.commit(__file__, fit_msg=fit_msg)

    device = torch.device(args.device)

    for k, v in args.__dict__.items():
        print(k, v)
    refresh_data = False
    if args.dataset == 'ppzh':
        datasets, vocabs, embeddings = load_zhPrivacyPolicyner(ppzh_ner_path, yangjie_rich_pretrain_unigram_path,
                                                               yangjie_rich_pretrain_bigram_path,
                                                               index_token=False,
                                                               )
    if args.dataset == 'ppzh':
        args.batch = args.batch
        args.gaz_dropout = args.gaz_dropout
        args.embed_dropout = args.embed_dropout
        args.output_dropout = args.output_dropout

    if args.gaz_dropout < 0:
        args.gaz_dropout = args.embed_dropout

    fitlog.set_log_dir('fitlog_log')
    fitlog.add_hyper(args)
    w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                                  _refresh=refresh_data)

    cache_name = os.path.join('cache', args.dataset + '_lattice')
    datasets, vocabs, embeddings = equip_chinese_ner_with_skip(datasets, vocabs, embeddings, w_list,
                                                               yangjie_rich_pretrain_word_path,
                                                               _refresh=refresh_data, _cache_fp=cache_name)
    print(datasets['train'][0])
    print('vocab info:')
    for k, v in vocabs.items():
        print('{}:{}'.format(k, len(v)))

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

    print(datasets['dev']['skips_l2r_word'][100])

    if args.model == 'lattice':
        model = LatticeLSTM_SeqLabel_V1(embeddings['char'], embeddings['bigram'], embeddings['word'],
                                        hidden_size=args.hidden, label_size=len(vocabs['label']), device=args.device,
                                        embed_dropout=args.embed_dropout, output_dropout=args.output_dropout,
                                        skip_batch_first=True, bidirectional=args.bi, debug=args.debug,
                                        skip_before_head=args.skip_before_head, use_bigram=args.use_bigram,
                                        gaz_dropout=args.gaz_dropout
                                        )
    elif args.model == 'lstm':
        model = LSTM_SeqLabel(embeddings['char'], embeddings['bigram'], embeddings['word'],
                              hidden_size=args.hidden, label_size=len(vocabs['label']), device=args.device,
                              bidirectional=args.bi,
                              embed_dropout=args.embed_dropout, output_dropout=args.output_dropout,
                              use_bigram=args.use_bigram)
    # kz
    # model = nn.DataParallel(model)
    # gpus = [0,1,2,3,4]
    # torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    # model = nn.DataParallel(model.cuda(),
    #                         device_ids=gpus,
    #                         output_device=gpus[0])
    #
    loss = LossInForward()
    encoding_type = 'bmeso'
    # if args.dataset == 'weibo':
    #     encoding_type = 'bio'
    # encoding_type = 'bio'
    print(encoding_type)
    # f1_metric = SpanFPreRecMetric(vocabs['label'], pred='pred', target='target', seq_len='seq_len',
    #                               encoding_type=encoding_type)
    f1_metric = SpanFPreRecMetric(vocabs['label'], pred='pred', target='target', seq_len='seq_len')
    acc_metric = AccuracyMetric(pred='pred', target='target', seq_len='seq_len')
    metrics = [f1_metric, acc_metric]

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    callbacks = [
        FitlogCallback({'test': datasets['test'], 'train': datasets['train']}),
        LRScheduler(lr_scheduler=LambdaLR(optimizer, lambda ep: 1 / (1 + 0.03) ** ep))
    ]
    print('label_vocab:{}\n{}'.format(len(vocabs['label']), vocabs['label'].idx2word))
    # kz
    trainer = Trainer(datasets['train'], model,
                      optimizer=optimizer,
                      loss=loss,
                      metrics=metrics,
                      dev_data=datasets['dev'],
                      device=device,
                      batch_size=args.batch,
                      n_epochs=args.epoch,
                      dev_batch_size=args.test_batch,
                      callbacks=callbacks,
                      update_every=4)
    # trainer = Trainer(datasets['train'], model,
    #                   optimizer=optimizer,
    #                   loss=loss,
    #                   metrics=metrics,
    #                   dev_data=datasets['dev'],
    #                   device=device,
    #                   batch_size=args.batch,
    #                   n_epochs=args.epoch,
    #                   dev_batch_size=args.test_batch,
    #                   callbacks=callbacks)

    trainer.train()
    #
    saver = ModelSaver("./cache/model_ckpt_100_2.pkl")
    saver.save_pytorch(model, param_only=False)
    #
    tester = Tester(datasets['test'], model, metrics=SpanFPreRecMetric(vocabs['label'], pred='pred', target='target', seq_len='seq_len',only_gross=False))
    tester.test()
    
    # if (args.predict):
    #     print("test1")
    #     model1 = ModelLoader.load_pytorch_model("./cache/model_ckpt_100_2.pkl")
    #     tester1 = Tester(datasets['test'], model1, metrics=SpanFPreRecMetric(vocabs['label'], pred='pred', target='target', seq_len='seq_len'))

    #     tester1.test()
