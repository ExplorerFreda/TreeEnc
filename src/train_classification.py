import argparse
import logging
import os

import math

from torch import nn, optim
from torch.nn.utils import clip_grad_norm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from src.data import Vocab
from src.models import SentClassModel
from src.utils import *

torch.save(args, args.save_dir + '/args.pt')

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


def train(args):
    word_vocab = Vocab.from_file(args.vocab_path, add_pad=True, add_unk=True)
    label_vocab = Vocab(dict([(i+1, i) for i in range(args.num_classes)]), add_pad=False, add_unk=False)
    train_dataset = SentenceDataset(
        data_path=args.data_prefix + '_' + 'train.json',
        word_vocab=word_vocab,
        label_vocab=label_vocab,
        max_length=args.max_length,
        lower=args.lower
    )
    valid_dataset = SentenceDataset(
        data_path=args.data_prefix + '_' + 'dev.json',
        word_vocab=word_vocab,
        label_vocab=label_vocab,
        max_length=args.max_length,
        lower=args.lower
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2,
        collate_fn=train_dataset.collate,
        pin_memory=True
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2,
        collate_fn=valid_dataset.collate,
        pin_memory=True
    )

    model = SentClassModel(
        num_classes=args.num_classes, num_words=len(word_vocab),
        word_dim=args.word_dim, hidden_dim=args.hidden_dim,
        clf_hidden_dim=args.clf_hidden_dim,
        clf_num_layers=args.clf_num_layers,
        use_leaf_rnn=args.leaf_rnn,
        use_batchnorm=args.batchnorm,
        pooling_method=args.pooling,
        dropout_prob=args.dropout,
        bidirectional=args.bidirectional,
        encoder_type=args.encoder_type
    )
    if args.load_checkpoint is not None:
        model.load_state_dict(torch.load(args.load_checkpoint))
    if args.glove:
        logger.info('Loading GloVe pretrained vectors...')
        glove_weight = load_glove(
            path=args.glove, vocab=word_vocab,
            init_weight=model.word_embedding.weight.data.numpy())
        glove_weight[word_vocab.pad_id] = 0
        model.word_embedding.weight.data.set_(torch.FloatTensor(glove_weight))
    if args.fix_word_embedding:
        logger.info('Will not update word embeddings')
        model.word_embedding.weight.requires_grad = False
    if torch.cuda.is_available():
        model = model.cuda()
    try:
        optimizer_class = getattr(optim, args.optimizer)
    except AttributeError:
        raise Exception('Optimizer {:s} not supported'.format(args.optimizer))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_class(lr=args.lr, params=params, weight_decay=args.l2reg)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()

    def run_iter(batch, is_training):
        model.train(is_training)
        sentences = wrap_with_variable(batch['sentences'], volatile=not is_training)
        lengths = wrap_with_variable(batch['lengths'], volatile=not is_training)
        labels = wrap_with_variable(batch['labels'], volatile=not is_training)
        tree_strs = None
        if args.encoder_type == 'gumbel':
            if is_training:
                logits = model(sentences, lengths)
            else:
                logits, sentence_info = model(sentences, lengths, return_select_masks=True)
                masks = sentence_info[-1]
                raw_sentences = batch['raw_sentences']
                tree_strs = get_tree_structures(raw_sentences, lengths, masks)
        elif args.encoder_type in ['parsing', 'balanced', 'left', 'right']:
            if args.encoder_type == 'parsing':
                tree_masks = batch['masks']
                for i, mask in enumerate(tree_masks):
                    tree_masks[i] = wrap_with_variable(mask, volatile=not is_training)
            elif args.encoder_type == 'balanced':
                tree_masks = generate_balance_masks(lengths.data.tolist())
            elif args.encoder_type == 'left':
                tree_masks = generate_left_branch_masks(lengths.data.tolist())
            elif args.encoder_type == 'right':
                tree_masks = generate_right_branch_masks(lengths.data.tolist())
            else:
                raise Exception('Encoder type {:s} not implemented.'.format(args.encoder_type))
            logits = model(sentences, lengths, tree_masks)
            raw_sentences = batch['raw_sentences']
            if not is_training:
                tree_strs = get_tree_structures(raw_sentences, lengths, tree_masks)
        elif args.encoder_type == 'lstm':
            logits = model(sentences, lengths)
        else:
            raise Exception('Encoder {:s} not implemented.'.format(args.encoder_type))
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(labels, label_pred).float().mean()
        loss = criterion(input=logits, target=labels)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(parameters=params, max_norm=5)
            optimizer.step()
        return loss, accuracy, tree_strs

    num_train_batches = len(train_loader)
    validate_every = num_train_batches // 10
    best_valid_accuracy = 0
    iter_count = 0
    logger.info('Validate every {:d} batches'.format(validate_every))
    for epoch_num in range(args.max_epoch):
        logger.info('Epoch {:d}: start'.format(epoch_num))
        avg_train_accuracy = avg_train_loss = 0
        for batch_iter, train_batch in enumerate(train_loader):
            if args.encoder_type == 'gumbel':
                if iter_count % args.anneal_temperature_every == 0:
                    rate = args.anneal_temperature_rate
                    new_temperature = max([0.5, math.exp(-rate * iter_count)])
                    model.encoder.gumbel_temperature = new_temperature
                    logger.info('Iter #{:d}: '
                                'Set Gumbel temperature to {:.4f}'.format(iter_count, new_temperature))
            train_loss, train_accuracy, _ = run_iter(batch=train_batch, is_training=True)
            avg_train_loss += unwrap_scalar_variable(train_loss)
            avg_train_accuracy += unwrap_scalar_variable(train_accuracy)
            iter_count += 1
            if (batch_iter + 1) % args.log_period == 0:
                avg_train_loss /= args.log_period
                avg_train_accuracy /= args.log_period
                logger.info('Epoch {:d}: {:d} batches, train loss = {:.4f}, train accuracy = {:.4f}'.format(
                    epoch_num, batch_iter + 1, avg_train_loss, avg_train_accuracy
                ))
                avg_train_loss = avg_train_accuracy = 0
            if (batch_iter + 1) % validate_every == 0:
                valid_loss_sum = valid_accuracy_sum = 0
                num_valid_batches = len(valid_loader)
                tree_strs = None
                for valid_batch in valid_loader:
                    if not tree_strs:
                        valid_loss, valid_accuracy, tree_strs = run_iter(batch=valid_batch, is_training=False)
                    else:
                        valid_loss, valid_accuracy, _ = run_iter(batch=valid_batch, is_training=False)
                    valid_loss_sum += unwrap_scalar_variable(valid_loss)
                    valid_accuracy_sum += unwrap_scalar_variable(valid_accuracy)
                valid_loss = valid_loss_sum / num_valid_batches
                valid_accuracy = valid_accuracy_sum / num_valid_batches
                scheduler.step(valid_accuracy)
                progress = epoch_num + batch_iter/num_train_batches
                logger.info('Epoch {:.2f}: '
                            'valid loss = {:.4f}, '
                            'valid accuracy = {:.4f}'.format(progress, valid_loss, valid_accuracy))
                if tree_strs is not None:
                    logger.info('Tree Example: {:s}'.format(tree_strs[0]))
                if valid_accuracy > best_valid_accuracy:
                    best_valid_accuracy = valid_accuracy
                    model_filename = ('model-{:.2f}-{:.4f}-{:.4f}.pkl'.format(
                        progress, valid_loss, valid_accuracy))
                    model_path = os.path.join(args.save_dir, model_filename)
                    torch.save(model.state_dict(), model_path)
                    model_path = os.path.join(args.save_dir, 'best_model.pkl')
                    torch.save(model.state_dict(), model_path)
                    print('Saved the new best model to {:s}'.format(model_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-prefix', type=str, required=True)
    parser.add_argument('--vocab-path', type=str, required=True)
    parser.add_argument('--encoder-type', type=str, required=True)
    parser.add_argument('--lower', default=False, action='store_true')
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--word-dim', type=int, default=300)
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--clf-hidden-dim', type=int, default=1024)
    parser.add_argument('--clf-num-layers', type=int, default=1)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--leaf-rnn', default=False, action='store_true')
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--pooling', type=str, default=None)
    parser.add_argument('--batchnorm', default=False, action='store_true')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--anneal-temperature-every', type=int, default=1e10)  # for Gumbel
    parser.add_argument('--anneal-temperature-rate', type=float, default=0)  # for Gumbel
    parser.add_argument('--glove', default=None)
    parser.add_argument('--fix-word-embedding', default=False, action='store_true')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-epoch', type=int, default=5)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--l2reg', type=float, default=0)
    parser.add_argument('--log-period', type=int, default=100)
    parser.add_argument('--load-checkpoint', default=None)
    parser.add_argument('--comp-method', type=str, default='dot_nd')
    args = parser.parse_args()

    # config log
    if os.path.exists(args.save_dir):
        os.system('rm -rf ' + args.save_dir)
    os.system('mkdir ' + args.save_dir)

    torch.save(args, args.save_dir + '/args.pt')
    handler = logging.FileHandler(args.save_dir + '/train.log', 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    # console log
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    train(args)


if __name__ == '__main__':
    main()
