import os

from models import *
from utils import *

from pathlib import Path
import argparse
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datasets import load_dataset
from collections import Counter
from conlleval import evaluate
from evaluation import calculate_metrics


def load_data(train_data_path=None, val_data_path=None):
    tags = ['Task', 'Method', 'Material']
    mapping = make_tag_lookup_table()
    print(mapping)
    train_path = Path('data/') / "scierc_train.txt"
    test_path = Path('data/') / "scierc_test.txt"
    with open(train_path, 'r') as f:
        tokens = []
        for line in f.readlines():
            terms = line.split('\t')
            num_terms = int(terms[0])
            tokens.append(terms[1:num_terms])

    all_tokens = sum(tokens, [])
    all_tokens_array = np.array(list(map(str.lower, all_tokens)))

    counter = Counter(all_tokens_array)
    print(len(counter))

    num_tags = len(mapping)
    vocab_size = 20000
    vocabulary = [token for token, count in counter.most_common(vocab_size - 2)]

    train_data = tf.data.TextLineDataset(train_path)
    val_data = tf.data.TextLineDataset(test_path)

    batch_size = 32
    train_dataset = (
        train_data.map(map_record_to_training_data)
            .map(lambda x, y: (lowercase_and_convert_to_ids(x, vocabulary), y))
            .padded_batch(batch_size)
    )
    val_dataset = (
        val_data.map(map_record_to_training_data)
            .map(lambda x, y: (lowercase_and_convert_to_ids(x, vocabulary), y))
            .padded_batch(batch_size)
    )

    # build model
    ner_model = NERModel(num_tags, vocab_size, embed_dim=32, num_heads=4, ff_dim=64)
    loss = CustomNonPaddingTokenLoss()
    ner_model.compile(optimizer="adam", loss=loss)
    ner_model.fit(train_dataset, epochs=100)

    # evaluate
    calculate_metrics(val_dataset, mapping, ner_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate NER model")
    parser.add_argument('--train',
                        metavar='training data path',
                        type=str,
                        help='path for the train dataset')
    parser.add_argument('--val',
                        metavar='validation data path',
                        type=str,
                        help='path for the val dataset')

    args = vars(parser.parse_args())
    train_data_path = args['train']
    val_data_path = args['val']
    load_data(train_data_path, val_data_path)
    # print(f"args train is {args['train']}")
    # print(f"args val is {args['val']}")