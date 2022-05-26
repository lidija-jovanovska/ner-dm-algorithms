import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# read dataset of format record = dict({'tokens': [], 'ner_tags: [])
# export one line per document -> num_tokens    tokens  token_ids

# usage
# from datasets import load_dataset
# conll_data = load_dataset("conll2003")
# os.mkdir("data")
# export_to_file("./data/conll_train.txt", conll_data["train"])
# export_to_file("./data/conll_val.txt", conll_data["validation"])

def export_to_file(export_file_path, data):
    with open(export_file_path, "w") as f:
        for record in data:
            ner_tags = record["ner_tags"]
            tokens = record["tokens"]
            if len(tokens) > 0:
                f.write(
                    str(len(tokens))
                    + "\t"
                    + "\t".join(tokens)
                    + "\t"
                    + "\t".join(map(str, ner_tags))
                    + "\n"
                )


# usage
# mapping = make_tag_lookup_table()

def make_tag_lookup_table(ner_labels=None):
    if ner_labels is None:
        ner_labels = ["PER", "ORG", "LOC", "MISC"]
    iob_labels = ["B", "I"]
    ner_labels = ner_labels
    all_labels = [(label1, label2) for label2 in ner_labels for label1 in iob_labels]
    all_labels = ["-".join([a, b]) for a, b in all_labels]
    all_labels = ["[PAD]", "O"] + all_labels
    return dict(zip(range(0, len(all_labels) + 1), all_labels))


#usage
# train_dataset = (
#     train_data.map(map_record_to_training_data)
#     .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
#     .padded_batch(batch_size)
# )

def map_record_to_training_data(record):
    record = tf.strings.split(record, sep="\t")
    length = tf.strings.to_number(record[0], out_type=tf.int32)
    tokens = record[1 : length + 1]
    tags = record[length + 1 :]
    tags = tf.strings.to_number(tags, out_type=tf.int64)
    tags += 1
    # tf.print(tokens, tags)
    return tokens, tags


def lowercase_and_convert_to_ids(tokens, vocabulary):
    tokens = tf.strings.lower(tokens)
    lookup_layer = keras.layers.StringLookup(
        vocabulary=vocabulary
    )
    return lookup_layer(tokens)

def tokenize_and_convert_to_ids(text):
    tokens = text.split()
    return lowercase_and_convert_to_ids(tokens)