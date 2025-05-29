from torch.utils.data import RandomSampler, DataLoader,Dataset,WeightedRandomSampler
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn.functional as F
import json
import os
from collections import Counter


def dataset(args):
    
    if not os.path.isdir(os.path.join(args.tokenizer_dataset_path, args.target)):
        text_path = args.dataset_path+args.target+'/'+'text.csv'
        train_index_path = args.dataset_path+args.target+'/'+'train.json'
        valid_index_path = args.dataset_path+args.target+'/'+'valid.json'
        test_index_path = args.dataset_path+args.target+'/'+'test.json'
        text_data = pd.read_csv(text_path)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_bert_name)
        train_data = data_load(text_data,train_index_path,args.target)
        print('train Tokenizer')
        train_data = build_dataset(args,train_data,tokenizer)
        valid_data = data_load(text_data,valid_index_path,args.target)
        print('valid Tokenizer')
        valid_data = build_dataset(args,valid_data,tokenizer)
        test_data = data_load(text_data,test_index_path,args.target)
        print('test Tokenizer')
        test_data = build_dataset(args,test_data,tokenizer)
        if not os.path.isdir(os.path.join(args.tokenizer_dataset_path, args.target)):
            os.makedirs(os.path.join(args.tokenizer_dataset_path, args.target))
        torch.save(train_data,os.path.join(args.tokenizer_dataset_path, args.target, 'train.pt'))
        torch.save(valid_data,os.path.join(args.tokenizer_dataset_path, args.target, 'valid.pt'))
        torch.save(test_data,os.path.join(args.tokenizer_dataset_path, args.target, 'test.pt'))
    else:
        train_data = torch.load(os.path.join(args.tokenizer_dataset_path, args.target, 'train.pt'))
        valid_data = torch.load(os.path.join(args.tokenizer_dataset_path, args.target, 'valid.pt'))
        test_data = torch.load(os.path.join(args.tokenizer_dataset_path, args.target, 'test.pt'))

    train_iter = DataLoader(train_data, batch_size=args.batch_size)
    dev_iter = DataLoader(valid_data, batch_size=args.batch_size)
    test_iter= DataLoader(test_data, batch_size=args.batch_size,shuffle=False)
    print('dataset finish')
    return train_iter,dev_iter,test_iter

def get_sampler(temp_data):
    total_samples = len(temp_data.data['labels'])
    count_0 = Counter(temp_data.data['labels'])[0]
    count_1 = Counter(temp_data.data['labels'])[1]
    count_2 = Counter(temp_data.data['labels'])[2]
    weight_0 = total_samples / (3 * count_0)
    weight_1 = total_samples / (3 * count_1)
    weight_2 = total_samples / (3 * count_2)

    weights = [weight_0 if label == 0 else weight_1 if label == 1 else weight_2 for label in temp_data.data['labels']]

    sampler = WeightedRandomSampler(weights, len(temp_data), replacement=True)

    return sampler

def data_load(text_data,path,target):
    conversations = []
    label = []
    targets = []
    index_data = json.load(open(path))
    # load text data
    for index_list in tqdm(index_data,desc='data_load'):
        # load text data
        conversation = ''
        for index in index_list['index']:
            if index == index_list['index'][-1]:
                conversation += text_data.loc[text_data.iloc[:,0] == index]['text'].values[0].lower()
            else:
                conversation += text_data.loc[text_data.iloc[:,0] == index]['text'].values[0].lower()
                conversation += ' [SEP] '
        words = conversation.split(' ')

        sep_count = 0
        if len(words) > 450:
            sep_count = words[0:len(words)-450].count('[SEP]')
            last_words = words[-450:]
            extracted_text = ' '.join(last_words)
            conversations.append(extracted_text)
        else:
            conversations.append(conversation)
        label.append(index_list['stance'].lower())
        targets.append(target)

    data={
        'data':conversations,
        'label':label,
        'targets':targets
    }
    return data


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data["input_ids"])
    def __getitem__(self, idx):

        sample = {
            "input_ids": self.data["input_ids"][idx],
            "attention_mask": self.data["attention_mask"][idx],
            "token_type_ids": self.data["token_type_ids"][idx],
            "labels": self.data["labels"][idx],

            "target_ids": self.data["target_ids"][idx],
            "target_attention_mask": self.data["target_attention_mask"][idx],
            "target_token_type_ids": self.data["target_token_type_ids"][idx],
        }

        return sample

def pad_sequences(sequences, pad_value=0):
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = [torch.tensor(seq + [pad_value] * (max_length - len(seq)),dtype=torch.long) for seq in sequences]
    return torch.stack(padded_sequences)


def build_dataset(args,data,tokenizer):
    label_map = {"none": 0,"favor": 1,"against": 2}

    labels = [label_map[label] for label in data['label']]
    train_inputs = tokenizer(data['data'], max_length=args.max_seq_len, padding='max_length', truncation=True, return_tensors='pt')
    target = tokenizer(data['targets'], max_length=10, padding='max_length', truncation=True, return_tensors='pt')
    
    for i in range(len(train_inputs["token_type_ids"])):
        sep_indexes = [i for i, token_id in enumerate(train_inputs["input_ids"][i]) if token_id == tokenizer.sep_token_id]
        ini =0
        temp = len(sep_indexes)%2
        for idx in sep_indexes:
            train_inputs["token_type_ids"][i][ini:idx+1] = temp
            temp = (temp + 1) % 2
            ini = idx+1

    dict_data = {
    "input_ids": train_inputs["input_ids"],
    "attention_mask": train_inputs["attention_mask"],
    "token_type_ids": train_inputs["token_type_ids"],
    "labels": labels,
    "target_ids": target["input_ids"],
    "target_attention_mask": target["attention_mask"],
    "target_token_type_ids": target["token_type_ids"],
    }


    dataset = CustomDataset(dict_data)
    return dataset