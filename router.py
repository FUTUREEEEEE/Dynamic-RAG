# %%
from preprocess import load_dataset
from datasets import Dataset
import argparse
import sys
import json
from utils import read_json,write_json,submit_data
import numpy as np 
import torch
sys.argv = ['']

def create_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--exp_name', type=str, default='cls', help='Experiment name')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--use_cwq', choices=['only', 'both', 'none'], default='none', help='Use ComplexWebQuestions dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--model_path', type=str, default='/apdcephfs_cq10/share_1567347/share_info/zivtang/data/.cache/distilbert-base-uncased', help='Model path')
    parser.add_argument('--load_best_model_at_end', type=bool, default=True, help='Should to set False when in dynamic setting')
    parser.add_argument('--dataset',type=str,default='webqsp')
    parser.add_argument('--llm_prefix', type=str, default=None, help='The prefix for the LLM')
    parser.add_argument('--sample_dataset',type=float,default=None)
    parser.add_argument('--train_method_index', type=str, default="{0:'RoG',1:'Decaf_fid_gen',2:'ChatKBQA_gen'}", help='String representation of the dictionary')
    parser.add_argument('--test_method_index', type=str, default="{0:'RoG',1:'Decaf_fid_gen',2:'ChatKBQA_gen'}", help='String representation of the dictionary')
    
    return parser



class LlamaQueryHandler:
    def __init__(self):
        # Define model path
        self.model_path = "/apdcephfs_cq10/share_1567347/share_info/llm_models/Llama-2-70b-chat-hf"
        from vllm import LLM, SamplingParams
        
        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            stop_token_ids="",
            max_tokens=4096,
            temperature=0.4,
            top_k=40,
            top_p=0.85,
            repetition_penalty=1.05
        )
        
        # Initialize the model
        self.model = LLM(
            self.model_path,
            tokenizer=self.model_path,
            tensor_parallel_size=8,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            dtype=torch.float16,
            enforce_eager=True
        )
        
    def query_llama(self, message_dicts):
        # Generate responses using the model
        outputs = self.model.generate(
            [i["message"] for i in message_dicts],
            self.sampling_params
        )
        
        # Update the message dictionaries with the responses
        for index, message_dict in enumerate(message_dicts):
            message_dicts[index]["llama2_response"] = outputs[index].outputs[0].text
        
        return message_dicts

llama2_prompt='''<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>

{user_message}[/INST]'''

llamaindexrouter_prompt = '''Please choose the best retrieval method for the following question: {question} Options:

If the question pertains to multiple correct answers, select "KG Query Language Retrieval".
If answering the question involves a reasoning process, select "LLM agent Retrieval".
For all other scenarios, select "Dense Retrieval".
{EXAMPLES OF SELECTION}:

{USER_QUERY}
'''


parser = create_parser()
user_args = parser.parse_args()

test_set,_ = load_dataset(train=False,dataset=user_args.dataset)
test_set = Dataset.from_list(test_set,split='test')
# handler = LlamaQueryHandler()

# %%
all_recall = []
all_hit = []
all_delay = []
delay_dict = {'RoG': 3, 'Decaf_fid_gen': 1, 'ChatKBQA_gen': 15}

# Run the evaluation block three times
for _ in range(3):
    message_dicts = []

    for i in range(len(test_set)):
        # prompt = llama2_prompt.format(system_prompt='', user_message=llamaindexrouter_prompt.format(question=test_set['RawQuestion'][i]))
        prompt = llamaindexrouter_prompt.format(question=test_set['RawQuestion'][i])
        message = [
                {"role": "user", "content": prompt},
            ]
        message_dicts.append({'message': message})

    # message_dicts = handler.query_llama(message_dicts)
    message_dicts = submit_data(message_dicts,model_name= "gpt-3.5-turbo")




    recall = []
    hit = []
    delay = []
    methods = ['RoG', 'Decaf_fid_gen', 'ChatKBQA_gen']
    for index, i in enumerate(message_dicts):
        if 'response' not in i:
            continue
        for method in methods:
            if method.lower() in i['response'].lower():
                hit.append(test_set[index][f'{method}_eval']['hit'])
                recall.append(test_set[index][f'{method}_eval']['recall'])
                delay.append(delay_dict[method])
                break
            


    # Store metrics from this run
    all_recall.append(np.mean(recall))
    all_hit.append(np.mean(hit))
    all_delay.append(np.mean(delay))

# Calculate means and standard deviations
mean_recall = np.mean(all_recall, axis=0) * 100
std_recall = np.std(all_recall, axis=0) * 100

mean_hit = np.mean(all_hit, axis=0) * 100
std_hit = np.std(all_hit, axis=0) * 100

mean_delay = np.mean(all_delay, axis=0)
std_delay = np.std(all_delay, axis=0)

# Print results in "mean ± std" format
print(f'Average Recall: {mean_recall:.2f} ± {std_recall:.2f}')
print(f'Average Hit: {mean_hit:.2f} ± {std_hit:.2f}')
print(f'Average Delay: {mean_delay:.2f} ± {std_delay:.2f}')
