import numpy as np
import torch
import ast
import time
import torch.nn as nn
from tqdm import tqdm
import datetime
import torch.optim as optim
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
)
from utils import *
import matplotlib.pyplot as plt
from preprocess import load_dataset
from torch.optim import AdamW
from transformers import get_scheduler
import concurrent.futures
from functools import partial

from collections import OrderedDict
import multiprocessing
import argparse

multiprocessing.set_start_method("spawn", force=True)

# torch.autograd.set_detect_anomaly(True)


def create_parser():
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add the arguments
    parser.add_argument("--num_arms", type=int, default=3, help="The string to process")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./results/", help="The string to process"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
    )
    parser.add_argument("--des", type=str, required=False, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--loss_policy", type=str, default="MSE")
    parser.add_argument("--action_policy", type=str, default="greedy")
    parser.add_argument("--delay_scale", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="webqsp")
    parser.add_argument("--sample_dataset", type=float, default=None)

    parser.add_argument("--test_weight_lr", type=float, default=1e-3)
    parser.add_argument("--test_warmup_ratio", type=float, default=0.02)
    parser.add_argument("--test_lr", type=float, default=3e-5)
    parser.add_argument(
        "--test_action_policy",
        choices=["greedy", "none", "decay_greedy"],
        default="greedy",
    )
    parser.add_argument("--test_loss_policy", default="MSE")
    parser.add_argument("--test_scheduler", type=str, default="linear")
    parser.add_argument("--fine_tune_last_layer", type=bool, default=True)

    parser.add_argument("--explore_rate", type=float, default=0.1)
    parser.add_argument("--weight_lr", type=float, default=0.005)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--use_cwq", choices=["both", "only"], default=None)
    parser.add_argument("--online_learning", action="store_true")

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
    parser.add_argument(
        "--llm_prefix", type=str, default=None, help="The prefix for the LLM"
    )

    return parser


def test_network(environ, net, tokenizer, args):
    net.eval()
    net.to(args.device)
    rewards = []
    actions = []
    delays = []
    recalls = []
    losses = []
    action_probs_list = []

    accumulative_regret = 0
    accumulative_regrets = []

    if args.online_learning:
        net.train()

        accumulation_steps = args.accumulation_steps  # 梯度累积步骤（等于batch_size）
        num_training_steps = args.epochs * len(environ)
        num_warmup_steps = int(args.test_warmup_ratio * num_training_steps)

        if not args.fine_tune_last_layer:
            optimizer = AdamW(
                [
                    {
                        "params": [net.ggc_weight, net.param1, net.param2, net.param3],
                        "lr": args.test_weight_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in net.named_parameters()
                            if n not in ["param1", "param2", "param3", "ggc_weight"]
                        ],
                        "lr": args.test_lr,
                    },
                ]
            )
        else:
            # Freeze all parameters except the classifier's
            for name, param in net.named_parameters():
                if (
                    "classifier" not in name
                    and "pre_classifier" not in name
                    and name not in ["param1", "param2", "param3", "ggc_weight"]
                ):
                    param.requires_grad = False

            # List of classifier layer parameters (these are the ones we want to fine-tune)
            classifier_params = [
                p
                for n, p in net.named_parameters()
                if "classifier" in n or "pre_classifier" in n
            ]

            # Set up optimizer only for the classifier layer
            optimizer = AdamW(
                [
                    {"params": classifier_params, "lr": args.test_lr},
                    {
                        "params": [net.ggc_weight, net.param1, net.param2, net.param3],
                        "lr": args.test_weight_lr,
                    },
                ]
            )

        lr_scheduler = get_scheduler(
            name=args.test_scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps // accumulation_steps,
        )

    else:
        torch.set_grad_enabled(False)  # replace with torch.no_grad()

    pbar = tqdm(total=len(environ), desc="Testing", dynamic_ncols=True)

    for e in range(1, len(environ) + 1):
        query = environ.get_state()
        rewards_pred = net(
            **tokenizer(
                query,
                max_length=256,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(args.device)
        )
        action_probas = softmax(
            rewards_pred.logits[0].detach().cpu().numpy().copy()[: args.num_arms]
        )  # only use

        if not args.online_learning:
            arm = np.argmax(action_probas)
        elif (
            args.test_action_policy == "greedy" and np.random.rand() < args.explore_rate
        ):
            arm = np.random.randint(0, args.num_arms)
        elif args.test_action_policy == "decay_greedy" and np.random.rand() < float(
            1 / np.log(e + 0.00001)
        ):
            arm = np.random.randint(0, args.num_arms)
        else:
            arm = np.argmax(action_probas)

        reward, recall = environ.choose_arm(arm)
        delay = environ.get_delay(arm)
        accumulative_regret += 1 - reward

        if args.online_learning:
            # update model
            net.train()
            args.loss_policy = args.test_loss_policy
            loss, reward = policy_gradient_loss(
                args, reward, arm, rewards_pred, delay, recall, net
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            losses.append(loss.item())

        if reward == -1:
            reward = 0

        rewards.append(reward)
        actions.append(arm)
        delays.append(delay)
        recalls.append(recall)
        action_probs_list.append(action_probas)
        accumulative_regrets.append(accumulative_regret)

        pbar.update(1)

    torch.set_grad_enabled(True)
    # count each action
    action_count = np.zeros(args.num_arms)
    for i in actions:
        action_count[i] += 1
    print(f"Action count : {action_count}")
    print(f"Sum delay : {np.sum(delays)}")

    return (
        np.array(rewards),
        actions,
        np.sum(delays),
        np.array(recalls),
        np.array(losses),
        np.array(action_probs_list),
        np.array(accumulative_regrets),
    )


def policy_gradient_loss(args, reward, arm, rewards_pred, delay, recall, net):
    policy = args.loss_policy
    if policy == "MSE":
        if reward == -1:
            reward = 0
        true_rewards = rewards_pred.logits[0].detach().cpu().numpy().copy()
        true_rewards[arm] = reward

        loss = nn.MSELoss()(rewards_pred.logits[0].cpu(), torch.Tensor(true_rewards))

    if policy == "MO_GGF":
        if reward == -1:
            reward = 0
        true_rewards = rewards_pred.logits[0].detach().cpu().numpy().copy()
        true_rewards[arm] = reward

        weight = torch.tensor([0.6, 0.3, 0.1])

        loss_1 = nn.MSELoss()(rewards_pred.logits[0].cpu(), torch.Tensor(true_rewards))
        loss_2 = nn.KLDivLoss()(
            F.log_softmax(rewards_pred.logits[0].cpu(), dim=0),
            torch.tensor(Environment.method_delay_ce_label),
        ).sigmoid()

        true_recall_rewards = rewards_pred.logits[0].detach().cpu().numpy().copy()
        true_recall_rewards[arm] = recall

        loss_3 = nn.MSELoss()(
            rewards_pred.logits[0].cpu(), torch.Tensor(true_recall_rewards)
        )

        loss = torch.cat((loss_1[None], loss_2[None], loss_3[None]))
        loss, _ = torch.sort(loss, descending=True)
        loss = weight @ loss

    if policy == "MO_GGF_adaptive":
        if reward == -1:
            reward = 0
        true_rewards = rewards_pred.logits[0].detach().cpu().numpy().copy()
        true_rewards[arm] = reward

        weight = torch.stack(
            [
                torch.sigmoid(net.param1),
                torch.sigmoid(net.param2),
                torch.sigmoid(net.param3),
            ]
        ).to(args.device)

        loss_1 = nn.MSELoss()(
            rewards_pred.logits[0], torch.Tensor(true_rewards).to(args.device)
        )
        loss_2 = (
            args.delay_scale
            * nn.KLDivLoss()(
                F.log_softmax(rewards_pred.logits[0], dim=0),
                torch.tensor(Environment.method_delay_ce_label).to(args.device),
            ).sigmoid()
        )
        # loss_2 = torch.zeros_like(loss_1).to(args.device)

        true_recall_rewards = rewards_pred.logits[0].detach().cpu().numpy().copy()
        true_recall_rewards[arm] = recall

        loss_3 = nn.MSELoss()(
            rewards_pred.logits[0], torch.Tensor(true_recall_rewards).to(args.device)
        )

        loss = torch.cat((loss_1[None], loss_2[None], loss_3[None]))
        loss, _ = torch.sort(loss, descending=True)
        weight, _ = torch.sort(weight, descending=True)
        loss = weight @ loss + 1.0 / torch.linalg.vector_norm(weight)

    if policy == "ACC_MO_GGF_adaptive":
        if reward == -1:
            reward = 0
        true_rewards = rewards_pred.logits[0].detach().cpu().numpy().copy()
        true_rewards[arm] = reward

        weight = torch.stack([torch.sigmoid(net.param1), torch.sigmoid(net.param2)]).to(
            args.device
        )

        loss_1 = nn.MSELoss()(
            rewards_pred.logits[0], torch.Tensor(true_rewards).to(args.device)
        )
        # loss_2 = args.delay_scale * nn.KLDivLoss()(F.log_softmax(rewards_pred.logits[0],dim=0),torch.tensor(Environment.method_delay_ce_label).to(args.device)).sigmoid()
        # loss_2 = torch.zeros_like(loss_1).to(args.device)

        true_recall_rewards = rewards_pred.logits[0].detach().cpu().numpy().copy()
        true_recall_rewards[arm] = recall

        loss_2 = nn.MSELoss()(
            rewards_pred.logits[0], torch.Tensor(true_recall_rewards).to(args.device)
        )

        loss = torch.cat((loss_1[None], loss_2[None]))
        loss, _ = torch.sort(loss, descending=True)
        weight, _ = torch.sort(weight, descending=True)
        loss = weight @ loss + 1.0 / torch.linalg.vector_norm(weight)

    if policy == "MO_adaptive":
        if reward == -1:
            reward = 0
        true_rewards = rewards_pred.logits[0].detach().cpu().numpy().copy()
        true_rewards[arm] = reward

        weight = torch.stack(
            [
                torch.sigmoid(net.param1),
                torch.sigmoid(net.param2),
                torch.sigmoid(net.param3),
            ]
        ).to(args.device)

        loss_1 = nn.MSELoss()(
            rewards_pred.logits[0], torch.Tensor(true_rewards).to(args.device)
        )
        loss_2 = nn.KLDivLoss()(
            F.log_softmax(rewards_pred.logits[0], dim=0),
            torch.tensor(Environment.method_delay_ce_label).to(args.device),
        ).sigmoid()

        true_recall_rewards = rewards_pred.logits[0].detach().cpu().numpy().copy()
        true_recall_rewards[arm] = recall

        loss_3 = nn.MSELoss()(
            rewards_pred.logits[0], torch.Tensor(true_recall_rewards).to(args.device)
        )

        loss = torch.cat((loss_1[None], loss_2[None], loss_3[None]))
        # loss,_ =  torch.sort(loss,descending=True)
        loss = weight @ loss + 1.0 / torch.linalg.vector_norm(weight)

    if policy == "MO_GGF_adaptive_test_policy":
        if reward == -1:
            reward = 0
        true_rewards = rewards_pred.logits[0].detach().cpu().numpy().copy()
        true_rewards[arm] = reward

        weight = torch.stack([torch.sigmoid(net.param1), torch.sigmoid(net.param2)]).to(
            args.device
        )

        loss_1 = nn.MSELoss()(
            rewards_pred.logits[0], torch.Tensor(true_rewards).to(args.device)
        )
        loss_2 = (
            args.delay_scale
            * nn.KLDivLoss()(
                F.log_softmax(rewards_pred.logits[0], dim=0),
                torch.tensor(Environment.method_delay_ce_label).to(args.device),
            ).sigmoid()
        )

        loss = torch.cat((loss_1[None], loss_2[None]))
        loss, _ = torch.sort(loss, descending=True)
        weight, _ = torch.sort(weight, descending=True)
        loss = weight @ loss + 1.0 / torch.linalg.vector_norm(weight)

    return loss, reward


def train_network(environ, net, tokenizer, args):
    # optimizer and scheduler

    accumulated_loss = 0.0  # 新增：用于累积损失
    accumulation_steps = args.accumulation_steps  # 梯度累积步骤（等于batch_size）

    num_training_steps = args.epochs * len(environ)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)

    # if 'MSE' in args.loss_policy:
    net.ggc_weight = nn.Parameter(torch.tensor(0.5))
    net.param1 = nn.Parameter(torch.tensor(0.5))
    net.param2 = nn.Parameter(torch.tensor(0.5))
    net.param3 = nn.Parameter(torch.tensor(0.5))

    # * give weight larger lr
    optimizer = AdamW(
        [
            {
                "params": [net.ggc_weight, net.param1, net.param2, net.param3],
                "lr": args.weight_lr,
            },
            {
                "params": [
                    p
                    for n, p in net.named_parameters()
                    if n not in ["param1", "param2", "param3", "ggc_weight"]
                ],
                "lr": args.lr,
            },
        ]
    )

    lr_scheduler = get_scheduler(
        name=args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps // accumulation_steps,
    )

    # * net init
    net.train()
    net.to(args.device)

    # * variables init
    rewards = []
    losses = []
    weight_changes = []
    action_probs_list = []
    cumulated_regret = 0
    cumulated_regrets = []

    pbar = tqdm(total=num_training_steps, desc="Training", dynamic_ncols=True)

    previous_weights = [param.data.clone() for param in net.parameters()]

    for e in range(1, num_training_steps + 1):
        query = environ.get_state()
        rewards_pred = net(
            **tokenizer(
                query,
                max_length=256,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(args.device)
        )
        action_probs = softmax(rewards_pred.logits[0].detach().cpu().numpy().copy())
        action_probs_list.append(action_probs)

        # * action policy
        if args.action_policy == "greedy" and np.random.rand() < args.explore_rate:
            arm = np.random.randint(0, args.num_arms)
        elif args.action_policy == "decay_greedy" and np.random.rand() < float(
            1 / np.log(e + 0.00001)
        ):
            arm = np.random.randint(0, args.num_arms)
        else:
            arm = np.argmax(action_probs)

        reward, recall = environ.choose_arm(arm)
        delay = environ.get_delay(arm)

        # rewards_pred.logits[0] = F.softmax(rewards_pred.logits[0],dim=0) # adding this will cause converge to one arm

        loss, reward = policy_gradient_loss(
            args, reward, arm, rewards_pred, delay, recall, net
        )

        loss = loss / accumulation_steps
        losses.append(loss.item())
        rewards.append(reward)
        cumulated_regret += 1 - reward
        cumulated_regrets.append(cumulated_regret)

        # * gradient accumulation
        loss.backward()  # 反向传播，但不立即更新权重
        accumulated_loss += loss.item()

        if e % accumulation_steps == 0 or e == num_training_steps:
            optimizer.step()  # 更新权重
            lr_scheduler.step()
            optimizer.zero_grad()  # 重置梯度

            pbar.set_postfix(
                {"loss": round(accumulated_loss / accumulation_steps, 2)}, refresh=True
            )
            accumulated_loss = 0.0  # 重置累积损失

            # 计算和记录权重的变化
            weight_change = [
                torch.norm(param.data - prev_param).item()
                for param, prev_param in zip(net.parameters(), previous_weights)
            ]
            weight_changes.append(weight_change)
            # 更新 previous_weights 以用于下一次迭代
            previous_weights = [param.data.clone() for param in net.parameters()]

        pbar.update(1)

    return (
        np.array(losses),
        np.array(rewards),
        np.array(action_probs_list),
        np.array(weight_changes),
        np.array(cumulated_regrets),
    )


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        "/apdcephfs_cq10/share_1567347/share_info/zivtang/data/.cache/distilbert-base-uncased"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "/apdcephfs_cq10/share_1567347/share_info/zivtang/data/.cache/distilbert-base-uncased",
        num_labels=args.num_arms,
    )

    return tokenizer, model


def one_experiment(args):
    fix_random_seed(args.seed)

    print(f"{time.ctime()}:loading dataset,seed {args.seed}")

    train_set, _ = load_dataset(train=True, dataset=args.dataset)
    if args.sample_dataset:
        train_set = train_set[: int(args.sample_dataset * len(train_set))]

    train_set = np.random.permutation(train_set)

    print(f"{time.ctime()}:loading env,seed {args.seed}")

    env = Environment(arms=args.num_arms, dataset=train_set)

    updateMethodIndex(
        method_index=args.train_method_index, llm_prefix=args.llm_prefix, env=env
    )

    print(f"{time.ctime()}:loading models,seed {args.seed}")

    tokenizer, model = load_model(args)

    print(f"{time.ctime()}:finish init model and env,seed {args.seed}")

    (
        train_losses,
        train_rewards,
        train_action_probs,
        train_weight_changes,
        train_cumulated_regrets,
    ) = train_network(env, model, tokenizer, args)

    test_set, test_set_info = load_dataset(train=False, dataset=args.dataset)

    if args.use_cwq == "both":
        cwq_test, _ = load_dataset(train=False, dataset="cwq")
        test_set.extend(cwq_test)
        print(f"load cwq test, len : {len(cwq_test)}")
    elif args.use_cwq == "only":
        test_set, test_set_info = load_dataset(train=False, dataset="cwq")
        print(f"load cwq test, len : {len(test_set)}")
    else:
        print(f"load normal test, len : {len(test_set)}")

    test_set = np.random.permutation(test_set)

    test_env = Environment(arms=args.num_arms, dataset=test_set)

    updateMethodIndex(
        method_index=args.test_method_index, llm_prefix=args.llm_prefix, env=test_env
    )

    print(F.sigmoid(model.param1), F.sigmoid(model.param2), F.sigmoid(model.param3))

    (
        test_rewards,
        test_actions,
        test_delays,
        test_recalls,
        test_losses,
        test_action_probs,
        test_cumulated_regrets,
    ) = test_network(test_env, model, tokenizer, args)

    return {
        "test_env": test_env,
        "train_losses": train_losses,
        "train_rewards": train_rewards,
        "train_action_probs": train_action_probs,
        "train_weight_changes": train_weight_changes,
        "test_set_info": test_set_info,
        "test_rewards": test_rewards,
        "test_actions": test_actions,
        "test_delays": test_delays,
        "test_recalls": test_recalls,
        "method_index": test_env._method_index,
        "test_losses": test_losses,
        "test_action_probs": test_action_probs,
        "train_cumulated_regrets": train_cumulated_regrets,
        "test_cumulated_regrets": test_cumulated_regrets,
    }


def run(seed, args, now):
    print(f"{time.ctime()}:starting experiment {seed}")

    args.seed = seed
    # if seed >50:
    args.device = select_gpu(5)  # random free gpu with 20 gb memory
    # else:
    #     args.device = (seed - 42)%8

    result = one_experiment(args)

    # * plt
    plt.figure(figsize=(12, 26), dpi=300)
    plt.subplot(7, 1, 1)
    for i in range(
        result["train_action_probs"].shape[1]
    ):  # Looping over the number of actions
        plt.plot(
            running_mean(result["train_action_probs"][:, i], 50),
            label=f'{result["method_index"][i]} Probability',
        )
    plt.legend()
    plt.title("Train Action Probabilities")

    # Third subplot for train loss
    plt.subplot(7, 1, 3)
    plt.plot(running_mean(result["train_losses"], 50), label="Train Loss")
    plt.plot(running_mean(result["test_losses"], 50), label="Test Loss")
    plt.legend()
    plt.title("Loss")

    # Fourth subplot for weight changes
    plt.subplot(7, 1, 4)
    plt.plot(result["train_weight_changes"])
    # plt.legend()
    plt.title("Train Weight Changes")

    # Fifth subplot for test action probabilities
    plt.subplot(7, 1, 2)
    for i in range(
        result["test_action_probs"].shape[1]
    ):  # Looping over the number of actions
        plt.plot(
            running_mean(result["test_action_probs"][:, i], 50),
            label=f'{result["method_index"][i]} Probability',
        )
    plt.legend()
    plt.title("Test Action Probabilities")

    # Second subplot for average reward
    plt.subplot(7, 1, 5)
    plt.plot(running_mean(result["train_rewards"], window=50), label="Train Reward")
    plt.plot(running_mean(result["test_rewards"], window=50), label="Test Reward")
    plt.legend()
    plt.title("Average Reward")

    # plot some text information np.mean(result["test_rewards"])
    plt.subplot(7, 1, 6)
    plt.plot(result["train_cumulated_regrets"], label="Train Cumulated Regret")
    plt.plot(result["test_cumulated_regrets"], label="Test Cumulated Regret")
    plt.legend()
    plt.title("Cumulated Regret")

    plt.subplot(7, 1, 7)
    plt.text(
        0.1,
        0.9,
        f"Average train reward : {np.mean(result['train_rewards']):.5f}",
        fontsize=12,
        transform=plt.gca().transAxes,
    )
    # plt.text(0.1, 0.8, f"Average train recall : {np.mean(result['train_recalls']):.5f}", fontsize=12, transform=plt.gca().transAxes)

    plt.text(
        0.1,
        0.7,
        f"Average test reward : {np.mean(result['test_rewards']):.5f}",
        fontsize=12,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.6,
        f"Average test recall : {np.mean(result['test_recalls']):.5f}",
        fontsize=12,
        transform=plt.gca().transAxes,
    )

    plt.tight_layout()

    # if args.debug:
    #     plt.savefig(f"results/{now}_{args.exp_name}_debug.png")
    # else:
    #     plt.savefig(os.path.join(args.out_dir,now + "_" + args.exp_name, f'_train_mab{seed}.png'))

    # Collect all necessary information in a dictionary

    return result


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if args.sample_dataset:
        assert args.dataset == "cwq", "sample_dataset only works with cwq dataset"

    args.num_arms = len(
        ast.literal_eval(args.train_method_index)
    )  # autoset number of arms
    # assert args.lr < 1e-4, "lr should be smaller than 1e-5"

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # if not args.debug:
    #     os.mkdir(os.path.join(args.out_dir,now + "_" + args.exp_name))
    #     # save current script
    #     os.system(f"cp {__file__} {args.out_dir + now + '_' + args.exp_name + '/'}")

    avg_test_rewards = []
    avg_train_rewards = []
    avg_test_delays = []
    avg_test_recalls = []

    if args.debug:
        seeds = args.seed
        results = [run(args.seed, args, now)]
    else:
        seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
        print(f"{time.ctime()}:launching {len(seeds)} experiments")
        partial_run = partial(run, args=args, now=now)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(partial_run, seeds))

    for result in results:
        avg_test_rewards.append(np.mean(result["test_rewards"]))
        avg_train_rewards.append(np.mean(result["train_rewards"]))
        avg_test_delays.append(result["test_delays"])
        if "test_recalls" in result and len(result["test_recalls"]) > 0:
            avg_test_recalls.append(np.mean(result["test_recalls"]))
        else:
            avg_test_recalls.append(0)

    best_hit = -1
    best_recall = -1
    for i in results[0]["method_index"].values():
        if results[0]["test_set_info"][i]["hit"] > best_hit:
            best_hit = results[0]["test_set_info"][i]["hit"]
            best_method = i
        if results[0]["test_set_info"][i]["recall"] > best_recall:
            best_recall = results[0]["test_set_info"][i]["recall"]
            best_recall_method = i

    print(f"Average test reward : {np.mean(avg_test_rewards)}")
    print(f"Average test recall : {np.mean(avg_test_recalls)}")
    print(
        f"Experiment average mean test delay : {np.mean(avg_test_delays)/results[0]['test_env'].dataset.shape[0]:.5f} ± {np.std(avg_test_delays)/results[0]['test_env'].dataset.shape[0]:.5f} "
    )


