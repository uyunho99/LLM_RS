import tqdm
import numpy as np
import torch
import os
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import random

class RecDataset(Dataset):
    def __init__(self, args, user_seq, valid_item_ids, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = []
        self.max_len = args.max_seq_length
        self.user_ids = []
        self.contrastive_learning = args.model_type.lower() in ['fearec', 'duorec']
        self.data_type = data_type
        self.valid_item_ids = valid_item_ids

        if self.data_type == 'train':
            for user, seq in enumerate(user_seq):
                input_ids = seq[-(self.max_len + 2):-2]  # train data는 맨 뒤 두개 떼고

                for i in range(len(input_ids)):
                    self.user_seq.append(input_ids[:i + 1])  # [1,2,3]시퀀스가 -> 1 / 1,2 / 1,2,3 이렇게 append
                    self.user_ids.append(user)  # 첫번째 루프 0번째 user (self.user_seq에는 0번째 유저의 시퀀스 조각들)

            total_sequences = len(self.user_seq)

            if total_sequences > 0:
                trash_ind = total_sequences % args.batch_size
                if trash_ind > 0 and total_sequences > args.batch_size:
                    self.user_seq = self.user_seq[:-trash_ind]
                    self.user_ids = self.user_ids[:-trash_ind]

                total_sequences_after_truncation = len(self.user_seq)

                if total_sequences_after_truncation == 0:
                    raise ValueError("All sequences were truncated, resulting in an empty dataset.")
                
        elif self.data_type == 'valid':
            for sequence in user_seq:
                self.user_seq.append(sequence[:-1])  # valid data는 맨 뒤 한개 떼고
        else:
            self.user_seq = user_seq  # test

        self.test_neg_items = test_neg_items

        if self.contrastive_learning and self.data_type == 'train':
            if os.path.exists(args.same_target_path):
                self.same_target_index = np.load(args.same_target_path, allow_pickle=True)
            else:
                self.same_target_index = self.get_same_target_index()
                self.same_target_index = np.array(self.same_target_index)
                np.save(args.same_target_path, self.same_target_index)

    def get_same_target_index(self):
        num_items = max([max(v) for v in self.user_seq]) + 2
        same_target_index = [[] for _ in range(num_items)]
        
        user_seq = self.user_seq[:]
        tmp_user_seq = []
        for i in tqdm.tqdm(range(1, num_items)):
            for j in range(len(user_seq)):
                if user_seq[j][-1] == i:
                    same_target_index[i].append(user_seq[j])
                else:
                    tmp_user_seq.append(user_seq[j])
            user_seq = tmp_user_seq
            tmp_user_seq = []
        return same_target_index

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        items = self.user_seq[index]
        input_ids = items[:-1]
        answer = items[-1]

        seq_set = set(items)
        neg_answer = neg_sample(seq_set, self.valid_item_ids)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.max_len:]
        assert len(input_ids) == self.max_len

        # Print the pos_id and neg_id
        # print(f"Input Sequence: {input_ids}")
        # print(f"Positive ID (pos_id): {answer}")
        # print(f"Negative ID (neg_id): {neg_answer}")

        if self.data_type in ['valid', 'test']:
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.zeros(0, dtype=torch.long), # not used
                torch.zeros(0, dtype=torch.long), # not used
            )

        elif self.contrastive_learning:
            sem_augs = self.same_target_index[answer]
            sem_aug = random.choice(sem_augs)
            keep_random = False
            for i in range(len(sem_augs)):
                if sem_augs[0] != sem_augs[i]:
                    keep_random = True

            while keep_random and sem_aug == items:
                sem_aug = random.choice(sem_augs)

            sem_aug = sem_aug[:-1]
            pad_len = self.max_len - len(sem_aug)
            sem_aug = [0] * pad_len + sem_aug
            sem_aug = sem_aug[-self.max_len:]
            assert len(sem_aug) == self.max_len

            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.tensor(sem_aug, dtype=torch.long)
            )

        else:
            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.zeros(0, dtype=torch.long), # not used
            )

        return cur_tensors


def neg_sample(item_set, valid_item_ids):
    item = random.choice(valid_item_ids)
    while item in item_set:
        item = random.choice(valid_item_ids)
    return item


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_rating_matrix(data_name, seq_dic, max_item):
    
    num_items = max_item + 1
    valid_rating_matrix = generate_rating_matrix_valid(seq_dic['user_seq'], seq_dic['num_users'], num_items)
    test_rating_matrix = generate_rating_matrix_test(seq_dic['user_seq'], seq_dic['num_users'], num_items)

    return valid_rating_matrix, test_rating_matrix

def get_user_seqs_and_max_item(data_file):
    lines = open(data_file).readlines()
    lines = lines[1:]
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split('	', 1)
        items = items.split()
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    return user_seq, max_item

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    num_users = len(lines)

    return user_seq, max_item, num_users

def get_seq_dic(args):

    args.data_file = args.data_dir + args.data_name + '.txt'
    user_seq, max_item, num_users = get_user_seqs(args.data_file)
    seq_dic = {'user_seq':user_seq, 'num_users':num_users }

    return seq_dic, max_item, num_users

def get_dataloder(args, seq_dic, valid_item_ids):
    train_dataset = RecDataset(args, seq_dic['user_seq'], valid_item_ids, data_type='train')
    print('train_dataset', len(train_dataset))
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    eval_dataset = RecDataset(args, seq_dic['user_seq'], valid_item_ids, data_type='valid')
    print('eval_dataset', len(eval_dataset))
    if len(eval_dataset) == 0:
        raise ValueError("Evaluation dataset is empty.")

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    test_dataset = RecDataset(args, seq_dic['user_seq'], valid_item_ids, data_type='test')
    print('test_dataset', len(test_dataset))
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty.")

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_dataloader, eval_dataloader, test_dataloader
