from data_loader import load_examples
from collections import defaultdict
from tqdm import tqdm
import torch
import seqeval.metrics


def evaluate(args, model, tokenizer, dataloader, labels_list, output_file=None):
    print("Start to evaluate the model...")
    all_predictions = defaultdict(dict)
    all_labels = defaultdict(dict)
    all_input_ids = defaultdict(dict)

    sample_idx = 0
    for batch in dataloader:
        model.eval()

        inputs = {k: v.to(args.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs['logits']

        for i in range(logits.size(0)):
            for j in range(logits.size(1)):
                all_predictions[sample_idx][j] = logits[i, j].detach().cpu().max(dim=0)
                all_labels[sample_idx][j] = inputs['labels'][i][j]
                all_input_ids[sample_idx][j] = inputs["input_ids"][i][j]
            sample_idx += 1

    assert len(all_labels) == len(all_predictions) == len(all_labels)

    final_labels = []
    final_predictions = []
    final_words = []

    for idx in range(sample_idx):
        predictions = all_predictions[idx]
        labels = all_labels[idx]
        input_ids = all_input_ids[idx]

        predictions_sentence = []
        labels_sentence = []
        words_sentence = []

        for pred, label, word_ids in zip(predictions.items(), labels.items(), input_ids.items()):
            idx, (logits, max_idx) = pred
            idx, label_idx = label
            idx, word_idx = word_ids

            if label_idx.item() == -100:
                continue
            predictions_sentence.append(labels_list[max_idx.item()])
            labels_sentence.append(labels_list[label_idx.item()])
            words_sentence.append(tokenizer.convert_ids_to_tokens(word_idx.item()))

        final_predictions.append(predictions_sentence)
        final_labels.append(labels_sentence)
        final_words.append(words_sentence)

    assert len(final_predictions) == len(final_labels)

    if output_file:
        with open(output_file, "w") as fp:
            for item in zip(final_words, final_labels, final_predictions):
                for word, label, pred in zip(item[0], item[1], item[2]):
                    fp.write(" ".join([word, label, pred]) + "\n")
                fp.write("\n")

    print(seqeval.metrics.classification_report(final_labels, final_predictions, digits=4))
    return dict(
        f1=seqeval.metrics.f1_score(final_labels, final_predictions),
        precision=seqeval.metrics.precision_score(final_labels, final_predictions),
        recall=seqeval.metrics.recall_score(final_labels, final_predictions),
    )


