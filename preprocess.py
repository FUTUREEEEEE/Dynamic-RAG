import json
import re
from utils import *
from eval import eval_f1,eval_acc,eval_hit
import numpy as np
from multiprocessing import Pool

# BGE refer to BGE retrieval results
methods_key_webqsp = [ 'RoG', 'RoG_retrieval', 'RoG_reasoning_paths','RoG_reasoning_paths_gen', 'ToG', 'BGE','BGE_gen',
                    #   'StructGPT', 'BGE',
              'Decaf_retrieval','Decaf_fid','Decaf_fid_gen','Decaf_fid_top1',
               'ChatKBQA_retrieval','ChatKBQA_gen','Decaf_fid_gen_phi2','ChatKBQA_gen_phi2','RoG_phi2',
               'Decaf_fid_gen_mistral','ChatKBQA_gen_mistral','RoG_mistral',
                'Decaf_fid_gen_phi3','ChatKBQA_gen_phi3','RoG_phi3',
                'Decaf_fid_gen_chatglm3','ChatKBQA_gen_chatglm3','RoG_chatglm3','StructGPT'
               ] # 'ToG', 'Dal_20','ToG_reasoning_chains', 'Dal_50', 'Dal_100',  'BGE_rerank','BGE_rerank_sentence',
methods_key_CWQ = ['RoG','Decaf_fid','Decaf_fid_gen','ChatKBQA_gen','ChatKBQA_retrieval','BGE','BGE_gen','Decaf_fid_top1'] # 'RoG_retrieval',,'Decaf_fid_gen','ChatKBQA_gen'

rag_base_dir = 'PATH_TO_RETRIEVAL/'
code_base = '.'


llm_names = ['phi-2','Mistral-7B-Instruct-v0.2']

def update_results_dict(results_dict, dataset, methods_key):
    for i in dataset:
        for method in methods_key:
            update_metric(results_dict[method], i[method+'_eval'])
                
    for method in methods_key:
        for key in results_dict[method].keys():
            results_dict[method][key] = round(sum(results_dict[method][key]) / float(len(results_dict[method][key])),5)

def eval(prediction, answer):  
    assert type(answer) == list 
          
    if not isinstance(prediction, list):
        prediction = prediction.split("\n")        

    f1_score, precision_score, recall_score = eval_f1(prediction, answer)
    # f1_list.append(f1_score)
    # precission_list.append(precision_score)
    # recall_list.append(recall_score)
    prediction_str = ' '.join(prediction)
    acc = eval_acc(prediction_str, answer)
    hit = eval_hit(prediction_str, answer)
    
    assert recall_score != 0 or hit == 0
    
    ret = {'acc': acc, 'hit': hit, 'f1': f1_score, 'precission': precision_score, 'recall': recall_score}
    return ret 
    # acc_list.append(acc)
    # hit_list.append(hit)
    f2.write(json.dumps({'id': id, 'prediction': prediction, 'ground_truth': answer, 'acc': acc, 'hit': hit, 'f1': f1_score, 'precission': precision_score, 'recall': recall_score}) + '\n')                    

def update_metric(pre, cur):
    for key in cur.keys():
        if key not in pre.keys():
            pre[key] = [cur[key]]
        pre[key].append(cur[key])
    return pre
    
def process_data(data):
    
    global methods_key_webqsp

    anwser = []
    for i in data['Parses']:
        assert not isinstance(i['TopicEntityName'], list)
        for j in i['Answers']:
            if j['AnswerType'] == 'Value':
                anwser.append(j['AnswerArgument'])
            else:
                assert j['AnswerType'] == 'Entity'
                anwser.append(j['EntityName'])
    
    # anwser.extend(data['additional_test_label'])
    # filter empty
    anwser = [lst for lst in anwser if lst]
    
    for method in methods_key_webqsp:
        prediction = data[method]
        if method == 'ToG_reasoning_chains':
            flattened_list = [item for sublist1 in prediction for sublist2 in sublist1 for item in sublist2]
            prediction = [item for sublist in flattened_list for item in sublist]
        ret = eval(prediction, anwser)
        data[method+'_eval'] = ret
    return data

def process_data_CWQ(data):
    global methods_key_CWQ
    
    anwser = data['answer']
    
    # filter empty
    anwser = [lst for lst in anwser if lst]
    
    for method in methods_key_CWQ:
        prediction = data[method]
        ret = eval(prediction, anwser)
        data[method+'_eval'] = ret
    return data
        
def load_dataset(train = True, reload = False, dataset = 'webqsp'):
    if dataset == 'webqsp':
        print('load webqsp')
        return load_dataset_webqsp(train, reload)
    elif dataset == 'cwq':
        print('load cwq')
        return load_dataset_CWQ(train, reload)

        
def load_dataset_webqsp(train = True, reload = False, datadir = 'data/processed_webqsp'):
    global methods_key_webqsp
    
    reasoning_paths_re = re.compile(r'Reasoning Paths:\n(.*?)\n\n', re.DOTALL)
    
    if train:
        
        if not reload and os.path.exists(os.path.join(datadir, 'train_set.json')):
            print('load train_set from cache')
            return json.load(open(os.path.join(datadir, 'train_set.json'))), json.load(open(os.path.join(datadir, 'train_set_results.json')))
        
        ori_train = json.load(open(rag_base_dir+'ChatKBQA/data/WebQSP/sexpr/WebQSP.train.expr.json'))

        ToG_train = load_jsonl(rag_base_dir+'ToG.new/ToG/ToG_webqsp_train.jsonl')

        RoG_train = load_jsonl(rag_base_dir+'reasoning-on-graphs/results/KGQA/RoG-webqsp/llama2-chat-hf/train/results_gen_rule_path_RoG-webqsp_RoG_train_predictions_3_False_jsonl/predictions.jsonl')

        RoG_retrieval_train = load_jsonl(rag_base_dir+'reasoning-on-graphs/results/KGQA/RoG-webqsp/RoG/train/_mnt_home_tangxiaqiang_code_gpt_RAG_reasoning-on-graphs_results_gen_rule_path_RoG-webqsp_RoG_train_predictions_3_False_jsonl/predictions.jsonl')
        
        RoG_reasoning_paths_gen = json.load(open(code_base+'results/generation/webqsp/train_llama2_new_rag_prompt/RoG_reasoning_paths_top-1.json'))
        
        BGE_train = json.load(open(rag_base_dir+'BGE/train.json'))

        # for name in llm_names:
        #     Rog_gen_

        
        # StructGPT_train = load_jsonl(rag_base_dir+'StructGPT/outputs/webqsp_train/output_wo_icl_v1.old.jsonl')
        
        # Dal_train = json.load(open(rag_base_dir+'decode-answer-logical-form/outputs/Retrieval/pyserini/search_results/QA_WebQSP_Freebase_BM25/train.json'))
        
        #load RAG generation data
        BGE_gen = json.load(open(code_base+'generation/webqsp/train_Llama-2-7b-chat-hf_rag_prompt_chat_templete/BGE_top-1.json'))
        
        
        # BGE_rerank_sentence_train = json.load(open(rag_base_dir+'BGE/train_stride_2reranked.json'))
        
        Decaf_retrieval = json.load(open(rag_base_dir+'decode-answer-logical-form/outputs/Retrieval/pyserini/search_results/QA_WebQSP_Freebase_BM25/train.json'))
        
        Decaf_fid = json.load(open(rag_base_dir+'decode-answer-logical-form/outputs/Reading/FiD.old/latest_p100_b10/final_output_train_fid_SPQA.json'))
        
        Decaf_fid_gen = json.load(open(code_base+'results/generation/webqsp/train_rog_model_select_prompt/Decaf_fid_top-1.json'))
        
        ChatKBQA_retrieval = json.load(open(rag_base_dir+'ChatKBQA.new/Reading/LLaMA2-7b/WebQSP_Freebase_NQ_lora_epoch100/checkpoint/checkpoint-8000/evaluation_beam/beam_test_top_k_predictions.json_gen_sexpr_results.json'))
        
        ChatKBQA_gen = json.load(open(code_base+'results/generation/webqsp/train_llama2_select_prompt/ChatKBQA_retrieval_top-1.json'))
        
        #phi2
        Decaf_fid_gen_phi2 = json.load(open(code_base+'generation/webqsp/train_phi-2_select_prompt/Decaf_fid_top-1.json')) 
        ChatKBQA_gen_phi2 = json.load(open(code_base+'generation/webqsp/train_phi-2_select_prompt/ChatKBQA_retrieval_top-1.json'))
        RoG_phi2 = json.load(open(code_base+'generation/webqsp/train_phi-2_select_prompt/RoG_retrieval_top-1.json'))

        # Mistral-7B-Instruct-v0.2
        Decaf_fid_gen_mistral = json.load(open(code_base+'generation/webqsp/train_Mistral-7B-Instruct-v0.2_select_prompt/Decaf_fid_top-1.json'))
        ChatKBQA_gen_mistral = json.load(open(code_base+'generation/webqsp/train_Mistral-7B-Instruct-v0.2_select_prompt/ChatKBQA_retrieval_top-1.json'))
        RoG_mistral = json.load(open(code_base+'generation/webqsp/train_Mistral-7B-Instruct-v0.2_select_prompt/RoG_retrieval_top-1.json'))

        # phi3
        Decaf_fid_gen_phi3 = json.load(open(code_base+'generation/webqsp/train_phi-3_select_prompt/Decaf_fid_top-1.json'))
        ChatKBQA_gen_phi3 = json.load(open(code_base+'generation/webqsp/train_phi-3_select_prompt/ChatKBQA_retrieval_top-1.json'))
        RoG_phi3 = json.load(open(code_base+'generation/webqsp/train_phi-3_rog_prompt2/RoG_retrieval_top-1.json'))

        #chatglm3-6b
        Decaf_fid_gen_chatglm3 = json.load(open(code_base+'generation/webqsp/train_chatglm3-6b_select_prompt/Decaf_fid_top-1.json'))
        ChatKBQA_gen_chatglm3 = json.load(open(code_base+'generation/webqsp/train_chatglm3-6b_select_prompt/ChatKBQA_retrieval_top-1.json'))
        RoG_chatglm3 = json.load(open(code_base+'generation/webqsp/train_chatglm3-6b_rog_prompt2/RoG_retrieval_top-1.json'))

        # to dict 
        ori_train = {item['QuestionId']: item for item in ori_train}
        ToG_train = {item['QuestionId']: item for item in ToG_train}
        RoG_train = {item['id']: item for item in RoG_train}
        RoG_retrieval_train = {item['id']: item for item in RoG_retrieval_train}
        RoG_reasoning_paths_gen = {item['QuestionId']: item for item in RoG_reasoning_paths_gen}
        BGE_train = {item['QuestionId']: item for item in BGE_train}
        # StructGPT_train = {item['ID']: item for item in StructGPT_train}
        # Dal_train = {item['QuestionId']: item for item in Dal_train}
        BGE_gen = {item['QuestionId']: item for item in BGE_gen}
        Decaf_retrieval = {item['QuestionId']: item for item in Decaf_retrieval}
        Decaf_fid = {k[:-3] :v for k,v in Decaf_fid.items() if 'QA' in k}
        Decaf_fid_gen = {item['QuestionId']: item for item in Decaf_fid_gen}
        ChatKBQA_retrieval = {item['qid']: item for item in ChatKBQA_retrieval}
        ChatKBQA_gen = {item['QuestionId']: item for item in ChatKBQA_gen}

        Decaf_fid_gen_phi2 = {item['QuestionId']: item for item in Decaf_fid_gen_phi2}
        ChatKBQA_gen_phi2 = {item['QuestionId']: item for item in ChatKBQA_gen_phi2}
        RoG_phi2 = {item['QuestionId']: item for item in RoG_phi2}

        Decaf_fid_gen_mistral = {item['QuestionId']: item for item in Decaf_fid_gen_mistral}
        ChatKBQA_gen_mistral = {item['QuestionId']: item for item in ChatKBQA_gen_mistral}
        RoG_mistral = {item['QuestionId']: item for item in RoG_mistral}

        Decaf_fid_gen_phi3 = {item['QuestionId']: item for item in Decaf_fid_gen_phi3}
        ChatKBQA_gen_phi3 = {item['QuestionId']: item for item in ChatKBQA_gen_phi3}
        RoG_phi3 = {item['QuestionId']: item for item in RoG_phi3}

        Decaf_fid_gen_chatglm3 = {item['QuestionId']: item for item in Decaf_fid_gen_chatglm3}
        ChatKBQA_gen_chatglm3 = {item['QuestionId']: item for item in ChatKBQA_gen_chatglm3}
        RoG_chatglm3 = {item['QuestionId']: item for item in RoG_chatglm3}
        
        
        train_set = []

        # print("num of ToG_trian: ", len(ToG_trian))
        print("num of RoG_train: ", len(RoG_train)) # TODO: debug why Rog has less data than ToG
        # print('num of BGE_train: ', len(BGE_train))
        # print('num of StructGPT_train: ', len(StructGPT_train))

        for key in ori_train.keys():
            if key not in RoG_train.keys() or key not in ChatKBQA_retrieval.keys() or key not in Decaf_fid_gen.keys():
                # print(key)
                continue
            sample = {}
            sample['RawQuestion'] = ori_train[key]['RawQuestion']   
            sample['QuestionId'] = key
            sample['Parses'] = ori_train[key]['Parses']
            sample['ToG'] = ToG_train[key]['results']
            # sample['ToG_reasoning_chains'] = ToG_trian[key]['reasoning_chains']
            sample['RoG'] = RoG_train[key]['prediction']
            sample['RoG_retrieval'] = RoG_retrieval_train[key]['input']
            match = reasoning_paths_re.search(sample['RoG_retrieval'])
            sample['RoG_reasoning_paths'] = match.group(1) if match else "" # * this one is clean reasoning paths
            sample['RoG_reasoning_paths_gen'] = RoG_reasoning_paths_gen[key]['RoG_reasoning_pathsgenerated_text']
            sample['BGE'] = BGE_train[key]['top_10_predictions']
            # # sample['StructGPT'] = StructGPT_train[key]['Prediction']
            # # sample['Dal_20'] = [i['text'] for i in Dal_train[key]['ctxs']][:20] # pick top 10
            # # sample['Dal_50'] = [i['text'] for i in Dal_train[key]['ctxs']][:50] # pick top 10
            # # sample['Dal_100'] = [i['text'] for i in Dal_train[key]['ctxs']][:100] # pick top 10
            sample['BGE_gen'] = BGE_gen[key]['BGEgenerated_text']
            # sample['BGE_rerank'] = BGE_rerank_train[key]['reranked_top_10_predictions'][:10]
            # sample['BGE_rerank_sentence'] = BGE_rerank_sentence_train[key]['reranked_top_10_predictions'][:10]
            sample['Decaf_retrieval'] = [i['text'] for i in Decaf_retrieval[key]['ctxs']][:100] # pick top 10
            sample['Decaf_fid'] = Decaf_fid[key]['predicted answers']
            sample['Decaf_fid_top1'] = Decaf_fid[key]['predicted answers'][0]

            sample['Decaf_fid_gen'] = Decaf_fid_gen[key]['Decaf_fidgenerated_text']
            
            sample['ChatKBQA_retrieval'] = [id2entity_name_or_type(i) for i in ChatKBQA_retrieval[key]['answer']]
            sample['ChatKBQA_gen'] = ChatKBQA_gen[key]['ChatKBQA_retrievalgenerated_text']

            sample['Decaf_fid_gen_phi2'] = Decaf_fid_gen_phi2[key]['Decaf_fidgenerated_text']
            sample['ChatKBQA_gen_phi2'] = ChatKBQA_gen_phi2[key]['ChatKBQA_retrievalgenerated_text']
            sample['RoG_phi2'] = RoG_phi2[key]['RoG_retrievalgenerated_text']

            sample['Decaf_fid_gen_mistral'] = Decaf_fid_gen_mistral[key]['Decaf_fidgenerated_text']
            sample['ChatKBQA_gen_mistral'] = ChatKBQA_gen_mistral[key]['ChatKBQA_retrievalgenerated_text']
            sample['RoG_mistral'] = RoG_mistral[key]['RoG_retrievalgenerated_text']

            sample['Decaf_fid_gen_phi3'] = Decaf_fid_gen_phi3[key]['Decaf_fidgenerated_text']
            sample['ChatKBQA_gen_phi3'] = ChatKBQA_gen_phi3[key]['ChatKBQA_retrievalgenerated_text']
            sample['RoG_phi3'] = RoG_phi3[key]['RoG_retrievalgenerated_text']

            sample['Decaf_fid_gen_chatglm3'] = Decaf_fid_gen_chatglm3[key]['Decaf_fidgenerated_text']
            sample['ChatKBQA_gen_chatglm3'] = ChatKBQA_gen_chatglm3[key]['ChatKBQA_retrievalgenerated_text']
            sample['RoG_chatglm3'] = RoG_chatglm3[key]['RoG_retrievalgenerated_text']
            
            
            train_set.extend([sample]) 
            
        print("num of train_set: ", len(train_set))

        # eval to get label

        results_dict = {method: {} for method in methods_key_webqsp}



        pool = Pool(10)  # Creates a Pool with 15 processes
        train_set = pool.map(process_data, train_set)  
        
        update_results_dict(results_dict, train_set, methods_key_webqsp)
        print(results_dict)
        
        # save 
        os.makedirs(datadir, exist_ok=True)
        json.dump(train_set, open(os.path.join(datadir, 'train_set.json'), 'w'))
        json.dump(results_dict, open(os.path.join(datadir, 'train_set_results.json'), 'w'))
        
        return train_set, results_dict
    
    else:
        
        if not reload and os.path.exists(os.path.join(datadir, 'test_set.json')):
            print('load test_set from cache')
            return json.load(open(os.path.join(datadir, 'test_set.json'))), json.load(open(os.path.join(datadir, 'test_set_results.json')))
        
        ori_test = json.load(open(rag_base_dir+'ChatKBQA/data/WebQSP/sexpr/WebQSP.test.expr.json'))

        # additional_test_label = json.load(open(rag_base_dir+'decode-answer-logical-form/data/tasks/QA/WebQSP/test.json'))

        ToG_test = load_jsonl(rag_base_dir+'ToG.new/ToG/ToG_webqsp_test.jsonl')
        RoG_test = load_jsonl(rag_base_dir+'reasoning-on-graphs/results/KGQA/RoG-webqsp/llama2-chat-hf/test/results_gen_rule_path_RoG-webqsp_RoG_test_predictions_3_False_jsonl/predictions.jsonl')
        RoG_retrieval_test = load_jsonl(rag_base_dir+'reasoning-on-graphs/results/KGQA/RoG-webqsp/RoG/test/_mnt_home_tangxiaqiang_code_gpt_RAG_reasoning-on-graphs_results_gen_rule_path_RoG-webqsp_RoG_test_predictions_3_False_jsonl/predictions.jsonl')
        RoG_reasoning_paths_gen = json.load(open(code_base+'results/generation/webqsp/test_llama2_new_rag_prompt/RoG_reasoning_paths_top-1.json'))
        StructGPT_test = load_jsonl(rag_base_dir+'StructGPT/outputs/webqsp_test/output_wo_icl_v1.old.jsonl')
        # Dal_test = json.load(open(rag_base_dir+'decode-answer-logical-form/outputs/Retrieval/pyserini/search_results/QA_WebQSP_Freebase_BM25/test.json'))
        BGE_test = json.load(open(rag_base_dir+'BGE/test.json'))
        BGE_rerank_test = json.load(open(rag_base_dir+'BGE/test_reranked.json'))
        BGE_rerank_sentence_test = json.load(open(rag_base_dir+'BGE/test_stride_2reranked.json'))
        
        BGE_gen_test = json.load(open(code_base+'results/generation/webqsp/test_llama2_rag_prompt/BGE_top10.json'))
        
        Decaf_retrieval = json.load(open(rag_base_dir+'decode-answer-logical-form/outputs/Retrieval/pyserini/search_results/QA_WebQSP_Freebase_BM25/test.json'))
        Decaf_fid = json.load(open(rag_base_dir+'decode-answer-logical-form/outputs/Retrieval/pyserini/search_results/QA_WebQSP_Freebase_BM25/final_output_test_fid_SPQA.json'))
        Decaf_fid_gen = json.load(open(code_base+'results/generation/webqsp/test_llama2_new_select_prompt/Decaf_fid_top-1.json'))
        
        ChatKBQA_retrieval = json.load(open(rag_base_dir+'ChatKBQA/Reading/LLaMA2-7b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/beam_test_top_k_predictions.json_gen_sexpr_results.json'))
        ChatKBQA_gen = json.load(open(code_base+'results/generation/webqsp/test_llama2_select_prompt/ChatKBQA_retrieval_top-1.json'))
        
        #phi2
        Decaf_fid_gen_phi2 = json.load(open(code_base+'generation/webqsp/test_phi-2_select_prompt/Decaf_fid_top-1.json'))
        ChatKBQA_gen_phi2 = json.load(open(code_base+'generation/webqsp/test_phi-2_select_prompt/ChatKBQA_retrieval_top-1.json'))
        RoG_phi2 = json.load(open(code_base+'generation/webqsp/test_phi-2_select_prompt/RoG_retrieval_top-1.json'))

        # Mistral-7B-Instruct-v0.2
        Decaf_fid_gen_mistral = json.load(open(code_base+'generation/webqsp/test_Mistral-7B-Instruct-v0.2_select_prompt/Decaf_fid_top-1.json'))
        ChatKBQA_gen_mistral = json.load(open(code_base+'generation/webqsp/test_Mistral-7B-Instruct-v0.2_select_prompt/ChatKBQA_retrieval_top-1.json'))
        RoG_mistral = json.load(open(code_base+'generation/webqsp/test_Mistral-7B-Instruct-v0.2_select_prompt/RoG_retrieval_top-1.json'))

        # phi3
        Decaf_fid_gen_phi3 = json.load(open(code_base+'generation/webqsp/test_phi-3_select_prompt/Decaf_fid_top-1.json'))
        ChatKBQA_gen_phi3 = json.load(open(code_base+'generation/webqsp/test_phi-3_select_prompt/ChatKBQA_retrieval_top-1.json'))
        RoG_phi3 = json.load(open(code_base+'generation/webqsp/test_phi-3_rog_prompt2/RoG_retrieval_top-1.json'))

        #chatglm3-6b
        Decaf_fid_gen_chatglm3 = json.load(open(code_base+'generation/webqsp/test_chatglm3-6b_select_prompt/Decaf_fid_top-1.json'))
        ChatKBQA_gen_chatglm3 = json.load(open(code_base+'generation/webqsp/test_chatglm3-6b_select_prompt/ChatKBQA_retrieval_top-1.json'))
        RoG_chatglm3 = json.load(open(code_base+'generation/webqsp/test_chatglm3-6b_rog_prompt2/RoG_retrieval_top-1.json'))

        BGE_gen = json.load(open(code_base+'generation/webqsp/test_Llama-2-7b-chat-hf_rag_prompt_chat_templete/BGE_top-1.json'))

        ori_test = {item['QuestionId']: item for item in ori_test}
        # additional_test_label = {item['QuestionId']: item for item in additional_test_label}
        ToG_test = {item['QuestionId']: item for item in ToG_test}
        RoG_test = {item['id']: item for item in RoG_test}
        RoG_retrieval_test = {item['id']: item for item in RoG_retrieval_test}
        RoG_reasoning_paths_gen = {item['QuestionId']: item for item in RoG_reasoning_paths_gen}
        StructGPT_test = {item['ID']: item for item in StructGPT_test}
        # Dal_test = {item['QuestionId']: item for item in Dal_test}
        BGE_test = {item['QuestionId']: item for item in BGE_test}
        BGE_gen_test = {item['QuestionId']: item for item in BGE_gen_test}
        Decaf_retrieval = {item['QuestionId']: item for item in Decaf_retrieval}
        # rewrite Decaf_fid keys
        Decaf_fid_test = {k[:-3] :v for k,v in Decaf_fid.items() if 'QA' in k}
        ChatKBQA_retrieval = {item['qid']: item for item in ChatKBQA_retrieval}
        ChatKBQA_gen = {item['QuestionId']: item for item in ChatKBQA_gen}
        Decaf_fid_gen = {item['QuestionId']: item for item in Decaf_fid_gen}

        Decaf_fid_gen_phi2 = {item['QuestionId']: item for item in Decaf_fid_gen_phi2}
        ChatKBQA_gen_phi2 = {item['QuestionId']: item for item in ChatKBQA_gen_phi2}
        RoG_phi2 = {item['QuestionId']: item for item in RoG_phi2}

        Decaf_fid_gen_mistral = {item['QuestionId']: item for item in Decaf_fid_gen_mistral}
        ChatKBQA_gen_mistral = {item['QuestionId']: item for item in ChatKBQA_gen_mistral}
        RoG_mistral = {item['QuestionId']: item for item in RoG_mistral}

        Decaf_fid_gen_phi3 = {item['QuestionId']: item for item in Decaf_fid_gen_phi3}
        ChatKBQA_gen_phi3 = {item['QuestionId']: item for item in ChatKBQA_gen_phi3}
        RoG_phi3 = {item['QuestionId']: item for item in RoG_phi3}

        Decaf_fid_gen_chatglm3 = {item['QuestionId']: item for item in Decaf_fid_gen_chatglm3}
        ChatKBQA_gen_chatglm3 = {item['QuestionId']: item for item in ChatKBQA_gen_chatglm3}
        RoG_chatglm3 = {item['QuestionId']: item for item in RoG_chatglm3}

        BGE_gen = {item['QuestionId']: item for item in BGE_gen}
        
        test_set = []
        
        print("num of ori_test: ", len(ori_test))
        
        for key in ori_test.keys():
            if key not in RoG_test.keys() or key not in ChatKBQA_retrieval.keys() or key not in Decaf_fid_gen.keys():
                # print(key)
                continue
            sample = {}
            sample['RawQuestion'] = ori_test[key]['RawQuestion']   
            sample['QuestionId'] = key
            sample['Parses'] = ori_test[key]['Parses']
            # sample['additional_test_label'] = [i['text'] for i in additional_test_label[key]['Answers']]

            sample['ToG'] = ToG_test[key]['results']
            # sample['ToG_reasoning_chains'] = ToG_test[key]['reasoning_chains']
            sample['RoG'] = RoG_test[key]['prediction']
            sample['RoG_retrieval'] = RoG_retrieval_test[key]['input']
            match = reasoning_paths_re.search(sample['RoG_retrieval'])
            sample['RoG_reasoning_paths'] = match.group(1) if match else ""
            sample['RoG_reasoning_paths_gen'] = RoG_reasoning_paths_gen[key]['RoG_reasoning_pathsgenerated_text']
            sample['StructGPT'] = StructGPT_test[key]['Prediction']
            sample['BGE'] = BGE_test[key]['top_10_predictions']
            # sample['Dal_20'] = [i['text'] for i in Dal_test[key]['ctxs']][:20] # pick top 10
            # sample['Dal_50'] = [i['text'] for i in Dal_test[key]['ctxs']][:50] # pick top 10
            # # sample['Dal_100'] = [i['text'] for i in Dal_test[key]['ctxs']][:100] # pick top 10
            # sample['BGE_rerank'] = BGE_rerank_test[key]['reranked_top_100_predictions'][:10]
            # sample['BGE_rerank_sentence'] = BGE_rerank_sentence_test[key]['reranked_top_10_predictions'][:10]
            sample['Decaf_retrieval'] = [i['text'] for i in Decaf_retrieval[key]['ctxs']][:100] # pick top 10
            sample['Decaf_fid'] = Decaf_fid_test[key]['predicted answers']
            sample['Decaf_fid_top1'] = Decaf_fid_test[key]['predicted answers'][0]
            # sample['BGE_gen'] = BGE_gen_test[key]['BGEgenerated_text']
            sample['ChatKBQA_retrieval'] = [id2entity_name_or_type(i) for i in ChatKBQA_retrieval[key]['answer']]
            sample['ChatKBQA_gen'] = ChatKBQA_gen[key]['ChatKBQA_retrievalgenerated_text']
            sample['ChatKBQA_sql'] = ChatKBQA_retrieval[key]['pred']
            
            sample['Decaf_fid_gen'] = Decaf_fid_gen[key]['Decaf_fidgenerated_text']


            sample['Decaf_fid_gen_phi2'] = Decaf_fid_gen_phi2[key]['Decaf_fidgenerated_text'] 
            sample['ChatKBQA_gen_phi2'] = ChatKBQA_gen_phi2[key]['ChatKBQA_retrievalgenerated_text'] 
            sample['RoG_phi2'] = RoG_phi2[key]['RoG_retrievalgenerated_text']

            sample['Decaf_fid_gen_mistral'] = Decaf_fid_gen_mistral[key]['Decaf_fidgenerated_text']
            sample['ChatKBQA_gen_mistral'] = ChatKBQA_gen_mistral[key]['ChatKBQA_retrievalgenerated_text']
            sample['RoG_mistral'] = RoG_mistral[key]['RoG_retrievalgenerated_text']

            sample['Decaf_fid_gen_phi3'] = Decaf_fid_gen_phi3[key]['Decaf_fidgenerated_text']
            sample['ChatKBQA_gen_phi3'] = ChatKBQA_gen_phi3[key]['ChatKBQA_retrievalgenerated_text']
            sample['RoG_phi3'] = RoG_phi3[key]['RoG_retrievalgenerated_text']

            sample['Decaf_fid_gen_chatglm3'] = Decaf_fid_gen_chatglm3[key]['Decaf_fidgenerated_text']
            sample['ChatKBQA_gen_chatglm3'] = ChatKBQA_gen_chatglm3[key]['ChatKBQA_retrievalgenerated_text']
            sample['RoG_chatglm3'] = RoG_chatglm3[key]['RoG_retrievalgenerated_text']

            sample['BGE_gen'] = BGE_gen[key]['BGEgenerated_text']
            # sample['ChatKBQA_retrieval'] = []
            # for i in ChatKBQA_retrieval[key]['answer']:
            #     sample['ChatKBQA_retrieval'].append(id2entity_name_or_type(i))
            
            test_set.extend([sample])
        
        print("num of test_set: ", len(test_set))
        
        # eval to get label
        
        results_dict = {method: {} for method in methods_key_webqsp}
        
        pool = Pool(20)  # Creates a Pool with 15 processes
        test_set = pool.map(process_data, test_set)  

                
        update_results_dict(results_dict, test_set, methods_key_webqsp)
                
        print(results_dict)
        
        #save
        os.makedirs(datadir, exist_ok=True)
        json.dump(test_set, open(os.path.join(datadir, 'test_set.json'), 'w'))
        json.dump(results_dict, open(os.path.join(datadir, 'test_set_results.json'), 'w'))
        
        return test_set, results_dict


        
        
def load_dataset_CWQ(train = True, reload = False, datadir = 'data/processed_cwq'):
    
    if train:
        if not reload and os.path.exists(os.path.join(datadir, 'train_set.json')):
            print('load train_set from cache')
            return json.load(open(os.path.join(datadir, 'train_set.json'))), json.load(open(os.path.join(datadir, 'train_set_results.json')))
        
        ori_dataset = json.load(open(code_base+'data/ComplexWebQuestions_train.json'))
        Decaf_fid = json.load(open(rag_base_dir+'decode-answer-logical-form/outputs/Reading/FiD/large_p100_b20/final_output_train_fid_SPQA.json'))
        RoG = load_jsonl(rag_base_dir+'reasoning-on-graphs/results/KGQA/RoG-cwq/llama2-chat-hf/train/results_gen_rule_path_RoG-cwq_RoG_train_predictions_3_False_jsonl/predictions.jsonl')
        ChatKBQA_retrieval = json.load(open(rag_base_dir+'ChatKBQA.new/Reading/LLaMA-13b/CWQ_Freebase_NQ_lora_epoch10/checkpoint/checkpoint-19000/evaluation_beam/beam_test_top_k_predictions.json_gen_sexpr_results.json'))
        ChatKBQA_gen = json.load(open(code_base+'generation/cwq/train_llama2_select_prompt/ChatKBQA_retrieval_top-1.json'))
        Decaf_fid_gen = json.load(open(code_base+'generation/cwq/train_llama2_select_prompt/Decaf_fid_top-1.json'))
        BGE = json.load(open(rag_base_dir+'BGE/output/CWQ_train.json'))
        BGE_gen = json.load(open(code_base+'generation/cwq/train_Llama-2-7b-chat-hf_rag_prompt_chat_templete/BGE_top10.json'))

        ori_dataset = {item['ID']: item for item in ori_dataset}
        Decaf_fid = {k[:-3] :v for k,v in Decaf_fid.items() if 'QA' in k}
        RoG = {item['id']: item for item in RoG}
        ChatKBQA_retrieval = {item['qid']: item for item in ChatKBQA_retrieval}
        ChatKBQA_gen = {item['QuestionId']: item for item in ChatKBQA_gen}
        Decaf_fid_gen = {item['QuestionId']: item for item in Decaf_fid_gen}
        BGE = {item['QuestionId']: item for item in BGE}
        BGE_gen = {item['QuestionId']: item for item in BGE_gen}

        train_set = []
        for key in ori_dataset.keys():
            if key not in ChatKBQA_retrieval.keys() or key not in RoG.keys() or key not in Decaf_fid.keys() or key not in BGE.keys() :
                continue
            sample = {}
            sample['answer'] = [i["answer"] for i in ori_dataset[key]['answers'] if i["answer"]]
            if len(sample['answer']) == 0:
                print(f"key: {key} has empty answer")
                continue
            sample['RawQuestion'] = ori_dataset[key]['question']
            sample['QuestionId'] = key
            sample['Decaf_fid'] = Decaf_fid[key]['predicted answers']
            sample['Decaf_fid_top1'] = Decaf_fid[key]['predicted answers'][0]
            sample['RoG'] = RoG[key]['prediction']
            sample['RoG_retrieval'] = RoG[key]['input']
            sample['ChatKBQA_sql'] = ChatKBQA_retrieval[key]['pred']
            sample['ChatKBQA_retrieval'] = [id2entity_name_or_type(i) for i in ChatKBQA_retrieval[key]['answer']]
            sample['Decaf_fid_gen'] = Decaf_fid_gen[key]['Decaf_fidgenerated_text']
            sample['ChatKBQA_gen'] = ChatKBQA_gen[key]['ChatKBQA_retrievalgenerated_text']
            sample['BGE'] = BGE[key]['top_10_predictions']
            sample['BGE_gen'] = BGE_gen[key]['BGEgenerated_text']

            train_set.extend([sample])

        print("num of cwq train_set: ", len(train_set))

        results_dict = {method: {} for method in methods_key_CWQ}
        pool = Pool(10)  # Creates a Pool with 15 processes
        train_set = pool.map(process_data_CWQ, train_set)

        update_results_dict(results_dict, train_set, methods_key_CWQ)

        print(results_dict)

        os.makedirs(datadir, exist_ok=True)
        json.dump(train_set, open(os.path.join(datadir, 'train_set.json'), 'w'))
        json.dump(results_dict, open(os.path.join(datadir, 'train_set_results.json'), 'w'))

        return train_set, results_dict
        

    else:
        if not reload and os.path.exists(os.path.join(datadir, 'test_set.json')):
            print('load test_set from cache')
            return json.load(open(os.path.join(datadir, 'test_set.json'))), json.load(open(os.path.join(datadir, 'test_set_results.json')))
        
        ori_dataset = json.load(open(code_base+'data/ComplexWebQuestions_test.json'))
        BGE = json.load(open(rag_base_dir+'BGE/output/CWQ_test.json'))
        BGE_gen = json.load(open(code_base+'generation/cwq/test_Llama-2-7b-chat-hf_rag_prompt_chat_templete/BGE_top10.json'))
        # BGE_llama7b = json.load(open(code_base+'results/generation/cwq/test_llama2_rag_prompt/BGE_top10.json'))
        # Decaf_fid = json.load(open(rag_base_dir+'decode-answer-logical-form/outputs/Retrieval/pyserini/search_results/QA_ComplexWebQuestions_BM25/final_output_test_fid_SPQA.json'))
        Decaf_fid = json.load(open(rag_base_dir+'decode-answer-logical-form/outputs/Reading/FiD/large_p100_b20/test_large_p100_b15/final_output_test_fid_SPQA.json'))
        RoG = load_jsonl(rag_base_dir+'reasoning-on-graphs/results/KGQA/RoG-cwq/llama2-chat-hf/test/results_gen_rule_path_cwq_RoG_test_predictions_3_False_jsonl/predictions.jsonl')
        RoG_retrieval = load_jsonl(rag_base_dir+'reasoning-on-graphs/results/KGQA/RoG-cwq/llama2-chat-hf/test/results_gen_rule_path_cwq_RoG_test_predictions_3_False_jsonl/predictions.jsonl')
        ChatKBQA_retrieval = json.load(open(rag_base_dir+'ChatKBQA.new/Reading/LLaMA-13b/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam/beam_test_top_k_predictions.json_gen_sexpr_results.json'))
        # Decaf_fid_gen = json.load(open(code_base+'results/generation/cwq/test_rog_model_select_prompt/Decaf_fid_top-1.json'))
        Decaf_fid_gen = json.load(open(code_base+'results/generation/cwq/test_llama2_new_select_prompt/Decaf_fid_top10.json'))

        ChatKBQA_gen = json.load(open(code_base+'results/generation/cwq/test_llama2_select_prompt/ChatKBQA_retrieval_top-1.json'))
        
        
        ori_test = {item['ID']: item for item in ori_dataset}      
        BGE = {item['QuestionId']: item for item in BGE}  
        BGE_gen = {item['QuestionId']: item for item in BGE_gen}
        # BGE_llama7b = {item['QuestionId']: item for item in BGE_llama7b}  
        Decaf_fid_test = {k[:-3] :v for k,v in Decaf_fid.items() if 'QA' in k}
        RoG = {item['id']: item for item in RoG}
        RoG_retrieval = {item['id']: item for item in RoG_retrieval}
        ChatKBQA_retrieval = {item['qid']: item for item in ChatKBQA_retrieval}
        Decaf_fid_gen = {item['QuestionId']: item for item in Decaf_fid_gen}
        ChatKBQA_gen = {item['QuestionId']: item for item in ChatKBQA_gen}
        
                
        test_set = []
        for key in ori_test.keys():
            if key not in ChatKBQA_gen.keys() or  key not in RoG.keys() or key not in Decaf_fid_gen.keys() :
                # fine which data is missing
                # if key not in ChatKBQA_gen.keys():
                #     print(f"key: {key} not in ChatKBQA_gen")
                # if key not in Decaf_fid_gen.keys():
                #     print(f"key: {key} not in Decaf_fid_gen")
                # if key not in RoG.keys():
                #     print(f"key: {key} not in RoG")
                
                continue
            
            sample = {}
            
            sample['answer'] = [i["answer"] for i in ori_test[key]['answers'] if i["answer"]]
            if len(sample['answer']) == 0:
                print(f"key: {key} has empty answer")
                continue
            
            sample['RawQuestion'] = ori_test[key]['question'] #*rename to RawQuestion
            sample['QuestionId'] = key

            

            
            sample['BGE'] = BGE[key]['top_10_predictions']
            sample['BGE_gen'] = BGE_gen[key]['BGEgenerated_text']
            # sample['BGE_llama7b'] = BGE_llama7b[key]['BGEgenerated_text']
            sample['Decaf_fid'] = Decaf_fid_test[key]['predicted answers']
            sample['Decaf_fid_top1'] = Decaf_fid_test[key]['predicted answers'][0]
            sample['RoG'] = RoG[key]['prediction']
            sample['RoG_retrieval'] = RoG_retrieval[key]['input']
            sample['ChatKBQA_retrieval'] = [id2entity_name_or_type(i) for i in ChatKBQA_retrieval[key]['answer']]
            sample['ChatKBQA_sql'] = ChatKBQA_retrieval[key]['pred']
            sample['Decaf_fid_gen'] = Decaf_fid_gen[key]['Decaf_fidgenerated_text']
            sample['ChatKBQA_gen'] = ChatKBQA_gen[key]['ChatKBQA_retrievalgenerated_text']
            
            test_set.extend([sample])
            
        print("num of test_set: ", len(test_set))
        
        results_dict = {method: {} for method in methods_key_CWQ}
        
        # for i in test_set:
            
        #     process_data_CWQ(i)
                
        pool = Pool(10)  # Creates a Pool with 15 processes
        test_set = pool.map(process_data_CWQ, test_set)
        
        update_results_dict(results_dict, test_set, methods_key_CWQ)
        
        print(results_dict)
        
        os.makedirs(datadir, exist_ok=True)
        json.dump(test_set, open(os.path.join(datadir, 'test_set.json'), 'w'))
        json.dump(results_dict, open(os.path.join(datadir, 'test_set_results.json'), 'w'))
        
        return test_set, results_dict
            

if __name__ == "__main__":
    from freebase_utils import id2entity_name_or_type
    
    # train_set, results_dict = load_dataset(train=True, reload=True)
    # test_set, results_dict = load_dataset(train=False, reload=True)


    # test_set, results_dict = load_dataset(train=True, reload=True,dataset="cwq")
    
    test_set, results_dict = load_dataset(train=False, reload=True,dataset="webqsp")
    
    
    
    
    
    # ToG_hit = []
    # RoG_hit = []
    
    # num_of_same = 0
    # num_ToG_can_and_RoG_can = 0
    # num_ToG_can_and_RoG_cannot = 0
    # num_ToG_cannot_and_RoG_can = 0
    # num_ToG_can = 0
    # num_RoG_can = 0
    
    # for i in train_set:
        
    #     if i['ToG_eval']['hit'] == 1:
    #         num_ToG_can += 1
    #     if i['RoG_eval']['hit'] == 1:
    #         num_RoG_can += 1
        
    #     if i['ToG_eval']['hit']== i['RoG_eval']['hit']:
    #         num_of_same += 1
    #     if i['ToG_eval']['hit'] == 1 and i['RoG_eval']['hit'] == 0:
    #         num_ToG_can_and_RoG_cannot += 1
    #     if i['ToG_eval']['hit'] == 0 and i['RoG_eval']['hit'] == 1:
    #         num_ToG_cannot_and_RoG_can += 1
    #     if i['ToG_eval']['hit'] == 1 and i['RoG_eval']['hit'] == 1:
    #         num_ToG_can_and_RoG_can += 1
        
    # print('num_ToG_can: ', num_ToG_can)
    # print('num_RoG_can: ', num_RoG_can)
    # print('num_ToG_can_and_RoG_cannot: ', num_ToG_can_and_RoG_cannot)
    # print('num_ToG_cannot_and_RoG_can: ', num_ToG_cannot_and_RoG_can)
    # print('num_ToG_can_and_RoG_can: ', num_ToG_can_and_RoG_can)
                
    # print('ration of same: ', num_of_same/len(train_set))
        
    
        
    # # compute intersection
    
    
    
    # print(train_set[0])
