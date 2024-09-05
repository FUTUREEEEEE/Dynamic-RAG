from utils import *
import json
import string


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_acc(prediction, answer):
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]


if __name__ == "__main__":
    dataset = json.load(
        open(
            "/mnt/home/tangxiaqiang/code/gpt/RAG/ChatKBQA/data/WebQSP/generation/merged/WebQSP_test.json"
        )
    )

    gt = json.load(
        open("/mnt/home/tangxiaqiang/code/gpt/RAG/ToG.new/eval/ToG_webqsp.json")
    )
    pred = json.load(
        open("/mnt/home/tangxiaqiang/code/gpt/Multi-Agent-RAG/data/ensemble/None.json")
    )
    RoG_pred = load_jsonl(
        "/mnt/home/tangxiaqiang/code/gpt/RAG/reasoning-on-graphs/results/KGQA/RoG-webqsp/RoG/test/_mnt_home_tangxiaqiang_code_gpt_RAG_reasoning-on-graphs_results_gen_rule_path_RoG-webqsp_RoG_test_predictions_3_False_jsonl/predictions.jsonl"
    )

    RoG_pred = {item["id"]: item for item in RoG_pred}

    pred = {item["query_id"]: item for item in pred}
    gt = {item["QuestionId"]: item for item in gt}
    dataset = {item["ID"]: item for item in dataset}

    num_query = 0
    num_hit = 0

    # eval gpt response
    for key in pred.keys():
        num_query += 1
        pred[key]["flag"] = False

        answer = [id2entity_name_or_type(i).lower() for i in dataset[key]["answer"]]
        response = pred[key]["prediction"].lower()
        for i in answer:
            if i.lower() in response.lower():
                pred[key]["flag"] = True
                break
        if not pred[key]["flag"]:
            response = response.split(" ")
            for i in response:
                if i in answer:
                    pred[key]["flag"] = True
                    break
        if not pred[key]["flag"]:
            pred[key]["flag"] = False
        else:
            num_hit += 1

        break
