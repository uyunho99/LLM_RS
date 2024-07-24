import torch
import os
import json
from langchain_community.embeddings import LlamaCppEmbeddings
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import json
from langchain_community.embeddings import LlamaCppEmbeddings
from torch.nn.utils.rnn import pad_sequence

embedding_vec_dict = None

def open_data_dict(file_path):
    with open(file_path, 'r') as file:
        contents_dict = json.load(file)
        contents_dict = {int(key): value for key, value in contents_dict.items()}
    return contents_dict

def get_embedding_for_text(input_text, max_length=None):
    embeddings = LlamaCppEmbeddings(model_path='./Llama-3-8B-Instruct-Gradient-1048k-Q4_K_M.gguf')
    if isinstance(input_text, list):
        embedding_output = [torch.tensor(embeddings.client.embed(text)) for text in input_text]
    elif isinstance(input_text, str):
        embedding_output = [torch.tensor(embeddings.client.embed(input_text))]
    elif isinstance(input_text, dict):
        embedding_output = {key: torch.tensor(embeddings.client.embed(value)) for key, value in input_text.items()}
    else:
        raise ValueError("입력은 문자열, 문자열 리스트 또는 문자열 딕셔너리여야 합니다.")
    
    # 패딩을 적용하여 모든 텐서의 길이를 동일하게 맞춤
    if max_length:
        if isinstance(embedding_output, list):
            embedding_output = [pad_or_truncate(emb, max_length) for emb in embedding_output]
        elif isinstance(embedding_output, dict):
            embedding_output = {key: pad_or_truncate(emb, max_length) for key, emb in embedding_output.items()}
    return embedding_output

def pad_or_truncate(tensor, max_length):
    if tensor.size(0) < max_length: # Paddding
        padding = torch.zeros(max_length - tensor.size(0), tensor.size(1))
        tensor = torch.cat((tensor, padding), dim=0)
    else: 
        tensor = tensor[:max_length] # Truncate
    return tensor

def build_embedding_vector_dict(args, hidden_size):
    global embedding_vec_dict 

    embedding_vec_dict_path = os.path.join(args.data_dir, 'embedding_vec_dict.pt') 
    
    # 이미 저장된 임베딩 딕셔너리가 있으면 불러옴
    if os.path.exists(embedding_vec_dict_path):
        embedding_vec_dict = torch.load(embedding_vec_dict_path)

    # 없으면 새로 생성
    else:
        args.data_file = args.data_dir + args.data_json_name + '.txt'
        contents_dict = open_data_dict(args.data_file)
    
        # Attribute dictionary의 description values를 임베딩 값으로 변환
        max_length = args.max_seq_length
        for key, value in contents_dict.items():
            embeddings = get_embedding_for_text(value, max_length=max_length)
            contents_dict[key] = torch.stack(embeddings) if isinstance(embeddings, list) else embeddings
            
        contents_dict[0] = torch.zeros(1, max_length, hidden_size)  # 패딩 0을 위한 제로 텐서 추가
        embedding_vec_dict = contents_dict
        torch.save(embedding_vec_dict, embedding_vec_dict_path)
        
    args.embedding_vec_dict = embedding_vec_dict
            
    return embedding_vec_dict

def convert_embedding_vector(hidden_size, batch_size, sequence, embedding_vec_dict, max_seq_length):
    if embedding_vec_dict is None:
        raise ValueError("embedding_vec_dict is None.")
    
    embedding_sequence = []

    if isinstance(sequence, torch.Tensor):
        sequence = sequence.tolist()

    if not isinstance(sequence, list):
        raise TypeError(f"Expected input sequence to be a list, but got {type(sequence)} instead.")

    for seq in sequence:
        if isinstance(seq, int):
            seq = [seq]
        emb_list = []
        for s in seq:
            # embedding_vec_dict에 키가 없으면 제로 텐서를 사용
            if s not in embedding_vec_dict:
                print(f"Warning: Key {s} not found in embedding_vec_dict. Using zero tensor as default.")
                emb = torch.zeros(max_seq_length, hidden_size)
            else:
                emb = embedding_vec_dict[s]
        
            if emb.dim() == 2: # emb가 [seq_len, hidden_size] 형태인 경우, batch 차원을 추가
                emb = emb.unsqueeze(0) 
            elif emb.dim() == 3 and emb.size(0) == 1: # emb가 [1, seq_len, hidden_size] 형태인 경우, 차원을 제거
                emb = emb.squeeze(0)
            emb_list.append(emb) # 각 시퀀스의 임베딩을 리스트에 추가

        # emb_list에 있는 모든 텐서를 concat해 하나의 텐서로 만듦(batch 차원을 기준으로)
        emb_tensor = torch.cat(emb_list, dim=0)

        # emb_tensor의 차원이 3이고, 두 번째 차원이 max_seq_length인 경우, (max_seq_length, hidden_size)로 변환
        if emb_tensor.dim() == 3 and emb_tensor.size(1) == max_seq_length:
            emb_tensor = emb_tensor.view(max_seq_length, hidden_size)
        
        # emb_tensor의 길이가 max_seq_length보다 작으면 패딩을 추가
        if emb_tensor.size(0) < max_seq_length:
            padding = torch.zeros(max_seq_length - emb_tensor.size(0), hidden_size)
            emb_tensor = torch.cat([emb_tensor, padding], dim=0)
        # emb_tensor의 길이가 max_seq_length보다 크면 잘라냄
        elif emb_tensor.size(0) > max_seq_length:
            emb_tensor = emb_tensor[:max_seq_length]
        embedding_sequence.append(emb_tensor)
    
    # embedding_sequence에 있는 모든 텐서를 concat해 하나의 텐서로 만듦(batch 차원을 기준으로)
    result = torch.stack(embedding_sequence, dim=0)
    
    return result