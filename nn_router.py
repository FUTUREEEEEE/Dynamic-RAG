from datasets import Dataset
from preprocess import load_dataset
import os
import ast
import argparse
import time
from utils import fix_random_seed, Environment, save_to_csv

from transformers import AutoTokenizer
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch

os.environ["WANDB_DISABLED"] = "true"


def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def create_parser():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--exp_name", type=str, default="cls", help="Experiment name")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--use_cwq",
        choices=["only", "both", "none"],
        default="none",
        help="Use ComplexWebQuestions dataset",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/apdcephfs_cq10/share_1567347/share_info/zivtang/data/.cache/distilbert-base-uncased",
        help="Model path",
    )
    parser.add_argument("--dataset", type=str, default="webqsp")
    parser.add_argument(
        "--llm_prefix", type=str, default=None, help="The prefix for the LLM"
    )
    parser.add_argument("--sample_dataset", type=float, default=None)
    parser.add_argument(
        "--train_method_index",
        type=str,
        default="{0:'RoG',1:'Decaf_fid_gen',2:'ChatKBQA_gen'}",
        help="String representation of the dictionary",
    )
    parser.add_argument(
        "--test_method_index",
        type=str,
        default="{0:'RoG',1:'Decaf_fid_gen',2:'ChatKBQA_gen'}",
        help="String representation of the dictionary",
    )

    return parser


def load_model(labels, id2label, label2id, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    return tokenizer, model


def main_loop(seed, user_args):
    fix_random_seed(seed)

    train_set, _ = load_dataset(dataset=user_args.dataset, train=True)
    if user_args.sample_dataset:
        train_set = train_set[: int(user_args.sample_dataset * len(train_set))]

    dataset = Dataset.from_list(train_set, split="train")

    if user_args.use_cwq == "only":
        test_set, _ = load_dataset(train=False, dataset="cwq")
        test_set = Dataset.from_list(test_set, split="test")
    elif user_args.use_cwq == "both":
        test_set, _ = load_dataset(dataset=user_args.dataset, train=False)
        cwq_test_set, _ = load_dataset(train=False, dataset="cwq")
        test_set = test_set + cwq_test_set
        test_set = Dataset.from_list(test_set, split="test")
    else:
        test_set, _ = load_dataset(dataset=user_args.dataset, train=False)
        test_set = Dataset.from_list(test_set, split="test")

    labels = list(ast.literal_eval(user_args.train_method_index).values())

    if user_args.llm_prefix is not None:
        # other llm experiments
        for i in labels:
            i += user_args.llm_prefix
    id2label = {}
    label2id = {}

    for index, i in enumerate(labels):
        id2label[index] = i
        label2id[i] = index

    tokenizer, model = load_model(labels, id2label, label2id, user_args.model_path)

    def preprocess_data(examples):
        # take a batch of texts
        text = examples["RawQuestion"]
        # encode them
        encoding = tokenizer(
            text, padding="max_length", truncation=True, max_length=128
        )
        # add labels
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(labels)))

        # fill numpy array
        for index, i in enumerate(labels_matrix):
            for key in labels:
                if examples[key + "_eval"][index]["hit"] == 1:
                    labels_matrix[index][label2id[key]] = 1

        recall_matrix = np.zeros((len(text), len(labels)))
        for index, i in enumerate(recall_matrix):
            for key in labels:
                recall_matrix[index][label2id[key]] = examples[key + "_eval"][index][
                    "recall"
                ]

        encoding["labels"] = labels_matrix.tolist()
        encoding["recall"] = recall_matrix.tolist()

        return encoding

    encoded_dataset = dataset.map(
        preprocess_data, batched=True, remove_columns=dataset.column_names
    )
    encoded_dataset.set_format("torch")

    encoded_dataset = encoded_dataset.train_test_split(test_size=0.1)
    train_set = encoded_dataset["train"]
    val_set = encoded_dataset["test"]

    labels = list(ast.literal_eval(user_args.train_method_index).values())
    if user_args.llm_prefix is not None:
        # other llm experiments
        for i in labels:
            i += user_args.llm_prefix

    for index, i in enumerate(labels):
        id2label[index] = i
        label2id[i] = index

    test_set = test_set.map(
        preprocess_data, batched=True, remove_columns=test_set.column_names
    )
    test_set.set_format("torch")

    metric_name = "f1"

    args = TrainingArguments(
        f"bert-finetuned-sem_eval-english",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=user_args.lr,
        per_device_train_batch_size=user_args.batch_size,
        per_device_eval_batch_size=user_args.batch_size,
        num_train_epochs=user_args.epochs,
        weight_decay=0.01,
        metric_for_best_model=metric_name,
        report_to=None,
        dataloader_drop_last=False,
        push_to_hub=False,
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.predict(test_set)

    lables = np.array(test_set["labels"])

    num_hit = 0
    sum_recall = 0
    sum_delay = 0

    actions = [0 for i in range(len(labels))]
    for index, i in enumerate(results.predictions):
        action_1 = np.argsort(i)[::-1][0]
        if test_set["labels"][index][action_1] == 1:
            num_hit += 1
        actions[action_1] += 1
        sum_recall += test_set["recall"][index][action_1]
        sum_delay += Environment._method_delay[action_1]
    return test_set, results, num_hit, sum_recall, sum_delay, actions


parser = create_parser()
user_args = parser.parse_args()

avg_hit = []
avg_recall = []
avg_delay = []

for seed in range(10):
    test_set, results, num_hit, sum_recall, sum_delay, actions = main_loop(
        seed=seed, user_args=user_args
    )

    avg_hit.append(num_hit / len(results.predictions))
    avg_recall.append(sum_recall / len(results.predictions))
    avg_delay.append(sum_delay / len(results.predictions))


print(f"Experiment avg_hit : {100*np.mean(avg_hit):.2f}  ± {100*np.std(avg_hit):.2f}")
print(f"Experiment avg_recall : {100*np.mean(avg_recall):.2f}  ± {100*np.std(avg_recall)}")
print(f"Experiment avg_delay : {np.mean(avg_delay):.2f}  ± {np.std(avg_delay)}")

