import argparse
import os
import torch
import json
from data_loader import load_examples
from models.ner_model import NERModel
from evaluate import evaluate
from trainer import Trainer
from transformers import (
    AutoTokenizer,
)
from transformers import WEIGHTS_NAME


def argument_parser():
    args = argparse.ArgumentParser()
    args.add_argument("--data_name",
                      default="conll2003",
                      type=str,
                      help="The input dataset name, such as conll2003")
    args.add_argument("--bert_model", default='bert-base-uncased', type=str,
                      help="Bert pre-trained model selected in the list: bert-base-uncased, "
                           "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    args.add_argument("--num_train_epochs",
                      default=2.0,
                      type=float,
                      help="Total number of training epochs to perform.")
    args.add_argument("--local_rank",
                      default=-1,
                      type=int,
                      help="local rank of gpu")
    args.add_argument("--debug",
                      default=False,
                      action='store_true',
                      help="Whether debug mode or not.")
    args.add_argument("--do_train",
                      default=False,
                      action='store_true',
                      help="Whether to run training.")
    args.add_argument("--no_cuda",
                      default=False,
                      action='store_true',
                      help="Whether to use CUDA when available")
    args.add_argument("--do_eval",
                      default=False,
                      action='store_true',
                      help="Whether to run eval on the dev set.")
    args.add_argument("--learning_rate",
                      default=5e-5,
                      type=float,
                      help="The initial learning rate for Adam.")
    args.add_argument('--gradient_accumulation_steps',
                      type=int,
                      default=1,
                      help="Number of updates steps to accumulate before performing a backward/update pass.")
    args.add_argument("--weight_decay",
                      default=0.01,
                      type=float,
                      help="Weight decay if we apply some.")
    args.add_argument("--adam_eps",
                      default=1e-6,
                      type=float,
                      help="Adam eps parameters")
    args.add_argument("--adam_b1",
                      default=1e-6,
                      type=float,
                      help="Adam b1 parameters")
    args.add_argument("--adam_b2",
                      default=1e-6,
                      type=float,
                      help="Adam b2 parameters")
    args.add_argument("--adam_correct_bias",
                      default=True,
                      type=bool,
                      help="Adam parameters")
    args.add_argument("--warmup_ratio",
                      default=0.06,
                      type=float,
                      help="Proportion of training to perform linear learning rate warmup for.")
    args.add_argument("--lr_schedule",
                      default="warmup_linear",
                      type=str,
                      help="Warmup schedule.")
    args.add_argument("--train_batch_size",
                      default=32,
                      type=int,
                      help="Total batch size for training.")
    args.add_argument("--max_grad_norm",
                      default=0.0,
                      type=float,
                      help="Max grad norm.")
    args.add_argument("--max_seq_length",
                      default=32,
                      type=int,
                      help="The maximum total input sequence length after WordPiece tokenization. \n"
                           "Sequences longer than this will be truncated, and sequences shorter \n"
                           "than this will be padded.")
    args.add_argument("--output_dir",
                      default="../outputs",
                      type=str,
                      help="The output directory where the model predictions and checkpoints will be written.")
    args.add_argument("--save_steps",
                      default=0,
                      type=int,
                      help="Save steps.")
    return args.parse_args()


def set_up_device(args):
    if args.no_cuda:
        device = torch.device("cpu")
        args.num_gpu = 0
    elif args.local_rank == -1:
        device = torch.device("cuda")
        args.num_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        args.num_gpu = 1
    args.device = device
    return device


def run():
    args = argument_parser()
    device = set_up_device(args)

    # set up output folder
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # load datasets
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    print("the device is: {}".format(args.device))
    if args.do_train:
        # load the training datasets
        train_datasets, labels_list = load_examples(args, tokenizer, 'train')

        model = NERModel(bert_model_name=args.bert_model, num_labels=len(labels_list))
        model.to(device)

        num_train_steps_per_epoch = len(train_datasets) // args.gradient_accumulation_steps
        num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)

        trainer = Trainer(args, model=model, dataloader=train_datasets, num_train_steps=num_train_steps)
        trainer.train()

    # save the model
    if args.do_train and args.local_rank in (0, -1):
        print("[RANK: {}]: Saving the model checkpoint to folder {}".format(args.local_rank, args.output_dir))
        torch.save(model.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))

    if args.local_rank not in (0, -1):
        return {}

    if args.do_eval:
        results = {}
        # load the test datasets
        dataloader, labels_list = load_examples(args, tokenizer, 'test')

        # load the model
        print("load the model...")
        torch.cuda.empty_cache()
        model = NERModel(bert_model_name=args.bert_model, num_labels=len(labels_list))
        model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME), map_location="cpu"))
        model.to(args.device)
        test_output_file = os.path.join(args.output_dir, "test_predictions.txt")
        results.update({f"test_{k}": v for k, v in evaluate(args, model, tokenizer, dataloader, labels_list, test_output_file).items()})

        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    run()
