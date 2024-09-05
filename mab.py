import numpy as np
import time
from collections import OrderedDict
import copy
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModel,
)
from utils import *
import matplotlib.pyplot as plt
from preprocess import load_dataset
from functools import partial
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

import pandas as pd
from sklearn.preprocessing import StandardScaler

from mab import algs
from contextualbandits.online import LinUCB
from NeuralUCB.learner_diag import NeuralUCBDiag

import gensim
from tqdm import tqdm
# from gensim.models.doc2vec import Doc2Vec,TaggedDocument


class neu_UCB(NeuralUCBDiag):
    def __init__(self, dim=100, lamdba=0.001, nu=1, hidden=100):
        super().__init__(dim, lamdba, nu, hidden)
        self.context = None

    def select(self, context):
        arm_select, nrm, sig, ave_rwd = super().select(context)
        self.context = context

        return arm_select, None

    def reward(self, reward):
        super().train(self.context, reward)

    def get_emb_model(self, dataset):
        contexts = [i["RawQuestion"] for i in dataset]
        tokens = []
        for i in contexts:
            token = gensim.utils.simple_preprocess(i)
            tokens.append(gensim.models.doc2vec.TaggedDocument(token, [i]))

        embed_model = gensim.models.doc2vec.Doc2Vec(
            vector_size=100, min_count=2, epochs=40
        )
        embed_model.build_vocab(tokens)
        embed_model.train(
            tokens, total_examples=embed_model.corpus_count, epochs=embed_model.epochs
        )

        self.embed_model = embed_model

        return embed_model

    def get_emb(self, context):
        return self.embed_model.infer_vector(gensim.utils.simple_preprocess(context))


class mab_LinUCB(LinUCB):
    # wrap to act like a no
    def __init__(self, arms, alpha=1.0):
        super().__init__(nchoices=arms)
        self.his_rewards = []
        self.his_actions = []
        self.his_contexts = []
        self.embed_method = "doc2vec"

    def get_emb_model(self, dataset):
        if self.embed_method == "doc2vec":
            contexts = [i["RawQuestion"] for i in dataset]
            tokens = []
            for i in contexts:
                token = gensim.utils.simple_preprocess(i)
                tokens.append(gensim.models.doc2vec.TaggedDocument(token, [i]))

            embed_model = gensim.models.doc2vec.Doc2Vec(
                vector_size=100, min_count=2, epochs=40
            )
            embed_model.build_vocab(tokens)
            embed_model.train(
                tokens,
                total_examples=embed_model.corpus_count,
                epochs=embed_model.epochs,
            )

            self.embed_model = embed_model

            return embed_model

        else:
            # use bert as embeding model
            contexts = [i["RawQuestion"] for i in dataset]

            tokenizer = AutoTokenizer.from_pretrained(
                "/apdcephfs_cq10/share_1567347/share_info/zivtang/data/.cache/distilbert-base-uncased"
            )

            model = AutoModel.from_pretrained(
                "/apdcephfs_cq10/share_1567347/share_info/zivtang/data/.cache/distilbert-base-uncased",
                num_labels=args.num_arms,
            )
            model.eval()

            context_embeddings = {}

            for context in contexts:
                inputs = tokenizer(
                    context, return_tensors="pt", padding=True, truncation=True
                )

                with torch.no_grad():
                    outputs = model(**inputs)

                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

                context_embeddings[context] = embeddings

            return context_embeddings

    def get_emb(self, context):
        if self.embed_method == "doc2vec":
            return self.embed_model.infer_vector(
                gensim.utils.simple_preprocess(context)
            )

        else:
            return self.embed_model[context]

    def select(self, context):
        # to np array
        context = np.array(context)

        action = super().predict(context).astype("uint8").item()

        self.his_contexts.append(context)
        self.his_actions.append(action)

        return action, None

    def reward(self, reward):
        self.his_rewards.append(reward)
        # super().fit(self.his_contexts[-1], self.his_actions[-1], self.his_rewards[-1])
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(self.his_contexts)
        # import pdb; pdb.set_trace()
        self.fit(
            X=np.stack(self.his_contexts),
            a=np.array(self.his_actions),
            r=np.array(self.his_rewards),
        )
        # TODO try partial_fit


class ModelProxy:
    # * enable the inference model by automatic restore the mab model
    def __init__(self, model):
        self._model = model
        self._saved_state = None

    def __getattr__(self, name):
        return getattr(self._model, name)

    def save_state(self):
        self._saved_state = copy.deepcopy(self._model.__dict__)

    def restore_state(self):
        self._model.__dict__.update(self._saved_state)


def create_parser():
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add the arguments
    parser.add_argument("--num_arms", type=int, default=2, help="The string to process")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )
    parser.add_argument("--mab", type=str, default="UCB1")
    # parser.add_argument('--epochs', type=int, default=1,)
    # parser.add_argument('--device', type=str, default='cuda:1',)
    # parser.add_argument('--out_dir', type=str,default='./results/',
    #                     help='The string to process')
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
    )
    parser.add_argument("--des", type=str, required=False, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="webqsp")
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
    parser.add_argument(
        "--llm_prefix", type=str, default=None, help="The prefix of the method"
    )

    # parser.add_argument('--debug',action='store_true')
    # parser.add_argument('--warmup_ratio',type=float,default=0)
    # parser.add_argument('--accumulation_steps',type=int,default=1)
    # parser.add_argument('--loss_policy',type=str,default='MSE')
    # parser.add_argument('--action_policy',type=str,default='greedy')
    # parser.add_argument('--explore_rate',type=float,default=0.1)
    # parser.add_argument('--weight_lr',type=float,default=1e-3)
    # parser.add_argument('--scheduler',type=str,default='linear')

    return parser


def load_model(args):
    # model = MAB(arms=np.arange(1, args.num_arms + 1),
    # learning_policy=LearningPolicy.UCB1(alpha=1.25),
    # neighborhood_policy=NeighborhoodPolicy.Radius(radius=5))
    if args.mab == "UCB1":
        model = algs.UCB1(args.num_arms)
    if args.mab == "UCBTuned":
        model = algs.UCBTuned(args.num_arms)
    if args.mab == "ThompsomSampling":
        model = algs.ThompsomSampling(args.num_arms)
    if args.mab == "LinUCB":
        model = mab_LinUCB(arms=args.num_arms)
    if args.mab == "NeuralUCB":
        model = neu_UCB(dim=100)

    return ModelProxy(model)


def train(model, env, args):
    # radius.fit(decisions=train_df['ad'], rewards=train_df['revenues'], contexts=train)

    num_training_steps = len(env)

    actions = np.zeros(args.num_arms)
    rewards = []

    tokenizer = AutoTokenizer.from_pretrained(
        "/apdcephfs/private_zivtang/data/.cache/distilbert-base-uncased"
    )

    for i in tqdm(range(num_training_steps)):
        if args.mab in ["LinUCB", "NeuralUCB"]:
            if i < 50 and args.mab == "LinUCB":
                # random init
                arm = np.random.choice(args.num_arms)
                reward, recall = env.choose_arm(arm)

                if reward == -1:
                    reward = 0

                # context = tokenizer(env.get_state(),max_length=50,truncation=True,padding='max_length',return_tensors='pt')
                # context = context['input_ids'][0].numpy()
                # pca_context = pca.transform(context.reshape(1,-1))[0]
                context = env.get_state()
                context = model.get_emb(context)

                # stardardize
                context = (
                    StandardScaler().fit_transform(context.reshape(-1, 1)).reshape(-1)
                )

                model.his_rewards.append(reward)
                model.his_actions.append(arm)
                model.his_contexts.append(context)

            else:
                context = env.get_state()
                context = model.get_emb(context)

                context = (
                    StandardScaler().fit_transform(context.reshape(-1, 1)).reshape(-1)
                )

                arm = model.select(context)[0]
                reward, recall = env.choose_arm(arm)

                if reward == -1:
                    reward = 0

                model.reward(reward)

        else:
            arm = model.select()[0]
            reward, recall = env.choose_arm(arm)
            if reward == -1:  #
                reward = 0
            else:
                model.reward(arm)

        actions[arm] += 1

        rewards.append(reward)

    ret = {
        "actions": actions,
        "rewards": rewards,
        "model": model,
        # if pca is not None, then return pca
    }

    return ret


def test(model, env, args):
    # radius.fit(decisions=train_df['ad'], rewards=train_df['revenues'], contexts=train)

    num_training_steps = len(env)

    actions = np.zeros(args.num_arms)
    rewards = []
    recalls = []
    delays = []

    tokenizer = AutoTokenizer.from_pretrained(
        "/apdcephfs/private_zivtang/data/.cache/distilbert-base-uncased"
    )

    for i in range(num_training_steps):
        if args.mab in ["LinUCB", "NeuralUCB"]:
            context = env.get_state()
            context = model.get_emb(context)

            context = StandardScaler().fit_transform(context.reshape(-1, 1)).reshape(-1)

            arm = model.select(context)[0]
            reward, recall = env.choose_arm(arm)

        else:
            model.save_state()
            arm = model.select()[0]
            model.restore_state()

            reward, recall = env.choose_arm(arm)

        if reward == -1:  #
            reward = 0

        # model.reward(reward)
        actions[arm] += 1
        recalls.append(recall)
        delays.append(env.get_delay(arm))

        rewards.append(reward)

    ret = {
        "actions": actions,
        "rewards": rewards,
        "model": model,
        "recalls": recalls,
        "delays": delays,
    }

    return ret


parser = create_parser()
args = parser.parse_args()

# fix_random_seed(args.seed)

# train_set,_ = load_dataset(train=True,dataset=args.dataset)
# train_set = np.random.permutation(train_set)

# if args.sample_dataset:
#     train_set = train_set[:int(args.sample_dataset*len(train_set))]


# model = load_model(args)
# env = Environment(arms=args.num_arms,dataset=train_set)

# ret_dict = train(model,env,args)

# #print
# print('Actions:', ret_dict['actions'])
# print('mean reward:', np.mean(ret_dict['rewards']))

# # plot

# plt.figure(figsize=(10, 5))

# plt.subplot(2, 2, 1)

# plt.plot(np.cumsum(ret_dict['rewards']))
# plt.title('Cumulative reward')
# plt.xlabel('Time')
# plt.ylabel('Cumulative reward')

# plt.subplot(2, 2, 2)
# plt.bar(np.arange(args.num_arms), ret_dict['actions'])
# plt.title('Actions')
# plt.xlabel('Arm')
# plt.ylabel('Number of times selected')


# test_set,_ = load_dataset(train=False,dataset=args.dataset)
# test_set = np.random.permutation(test_set)
# test_env = Environment(arms=args.num_arms,dataset=test_set)

# test_ret_dict = test(model,test_env,args,ret_dict['embed_model'] if 'embed_model' in ret_dict else None)

# print('Test Actions:', test_ret_dict['actions'])
# print('Test mean reward:', np.mean(test_ret_dict['rewards']))
# print('Test mean recall:', np.mean(test_ret_dict['recalls']))
# print('Test sum delay:', np.sum(test_ret_dict['delays']))


# plt.subplot(2, 2, 3)
# plt.plot(np.cumsum(test_ret_dict['rewards']))
# plt.title('Test Cumulative reward')
# plt.xlabel('Time')
# plt.ylabel('Cumulative reward')

# plt.subplot(2, 2, 4)
# plt.bar(np.arange(args.num_arms), test_ret_dict['actions'])
# plt.title('Test Actions')
# plt.xlabel('Arm')
# plt.ylabel('Number of times selected')

# plt.tight_layout()


# plt.savefig(f'./results/{args.exp_name}.png')

# run 10 times to get the average result
avg_test_rewards = []
avg_test_recalls = []
avg_test_delays = []

train_set, _ = load_dataset(train=True, dataset=args.dataset)
if args.sample_dataset:
    train_set = train_set[: int(args.sample_dataset * len(train_set))]
test_set, _ = load_dataset(train=False, dataset=args.dataset)

if args.mab in ["LinUCB", "NeuralUCB"]:
    model = load_model(args)
    embed_model = model.get_emb_model(train_set + test_set)

for i in range(3):
    model = load_model(args)
    if args.mab in ["LinUCB", "NeuralUCB"]:
        model._model.embed_model = embed_model
    train_set = np.random.permutation(train_set)

    env = Environment(arms=args.num_arms, dataset=train_set)
    updateMethodIndex(
        method_index=args.train_method_index, llm_prefix=args.llm_prefix, env=env
    )

    ret_dict = train(model, env, args)

    test_set = np.random.permutation(test_set)

    test_env = Environment(arms=args.num_arms, dataset=test_set)
    updateMethodIndex(
        method_index=args.test_method_index, llm_prefix=args.llm_prefix, env=test_env
    )

    test_ret_dict = test(model, test_env, args)

    print(f"avg reward {100*np.mean(test_ret_dict['rewards']):.3f}")
    print(f"avg recall {100*np.mean(test_ret_dict['recalls']):.3f}")
    print(f"avg delay {np.sum(test_ret_dict['delays'])/len(test_set) :.3f}")

    avg_test_rewards.append(np.mean(test_ret_dict["rewards"]))
    avg_test_recalls.append(np.mean(test_ret_dict["recalls"]))
    avg_test_delays.append(np.sum(test_ret_dict["delays"]))


print(
    f"avg reward {100*np.mean(avg_test_rewards):.3f}±{100*np.std(avg_test_rewards):.3f}"
)
print(
    f"avg recall {100*np.mean(avg_test_recalls):.3f}±{100*np.std(avg_test_recalls):.3f}"
)
print(
    f"avg delay {np.mean(avg_test_delays)/len(test_set) :.3f}±{np.std(avg_test_delays)/len(test_set):.3f}"
)


new_data_row = OrderedDict(
    {
        "time": time.ctime(),
        "exp_name": args.exp_name,
        "Test Accuracy": f"{100*np.mean(avg_test_rewards):.2f} ± {100*np.std(avg_test_rewards):.2f}",
        "Test Recall": f"{100*np.mean(avg_test_recalls):.2f} ± {100*np.std(avg_test_recalls):.2f}",
        "Test Retrieval Delay": f"{np.mean(avg_test_delays)/len(test_set):.2f} ± {np.std(avg_test_delays)/len(test_set):.2f} ",
        "acc": f"{100*np.mean(avg_test_rewards):.2f}",
        "recall": f"{100*np.mean(avg_test_recalls):.2f}",
        "delay": f"{np.mean(avg_test_delays)/len(test_set):.2f}",
        "args": str(args),
    }
)


csv_file_path = "./results/log.csv"
save_to_csv(new_data_row, csv_file_path)
