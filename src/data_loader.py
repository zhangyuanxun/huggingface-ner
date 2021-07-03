import torch
from datasets import load_dataset, load_metric
from torch.utils.data import TensorDataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler


def convert_to_features(examples, tokenizer, args):
    features = tokenizer(examples["tokens"], truncation=True,
                         padding="max_length", max_length=args.max_seq_length, is_split_into_words=True)
    labels = []

    for i, label in enumerate(examples["ner_tags"]):
        word_idx = features.word_ids(batch_index=i)
        label_ids = []

        for wid in word_idx:
            if wid is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[wid])
        labels.append(label_ids)

    features["labels"] = labels
    return features


def load_examples(args, tokenizer, datatype):
    if args.local_rank not in (-1, 0) and datatype == "train":
        torch.distributed.barrier()

    if args.data_name == "conll2003":
        examples = load_dataset("conll2003")
        labels = examples['train'].features["ner_tags"].feature.names

    features = examples.map(lambda x: convert_to_features(x, tokenizer, args), batched=True)

    if datatype == 'train':
        sample_datasets = features['train']
    elif datatype == 'test':
        sample_datasets = features['test']
    elif datatype == 'dev':
        sample_datasets = features['validation']

    if args.debug and args.do_train:
        sample_datasets = sample_datasets.select(range(50))

    print("The {} size of datasets is loaded.".format(sample_datasets.num_rows))

    features_cols = ['input_ids', 'attention_mask', 'labels']
    remove_cols = [col for col in sample_datasets.column_names if col not in features_cols]
    sample_datasets = sample_datasets.remove_columns(remove_cols)
    sample_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    if datatype == 'train':
        sampler = RandomSampler(sample_datasets) if args.local_rank == -1 else DistributedSampler(sample_datasets)
        dataloader = torch.utils.data.DataLoader(sample_datasets, sampler=sampler, batch_size=args.train_batch_size)
    else:
        dataloader = torch.utils.data.DataLoader(sample_datasets, batch_size=args.train_batch_size)

    return dataloader, labels

