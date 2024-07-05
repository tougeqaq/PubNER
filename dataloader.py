import os
import json
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BartForConditionalGeneration


# 为什么不采用BartTokenizer
class MyDataset(Dataset):
    def __init__(self, config, is_train_set=True):
        filename = config.trainset_path if is_train_set else config.testset_path
        datas = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                random_value = random.random()
                if random_value < config.get_data_scale:
                    datas.append(json.loads(line.strip()))
                else:
                    continue

        self.input_texts = [data['text'] for data in datas]  # input_text = "text"
        self.targets = [data['answer'] for data in datas]  # targets = ['answer1','answer2',...]

        self.tokenizer = config.tokenizer
        self.model = config.model

    def __getitem__(self, index):
        return self.input_texts[index], self.targets[index]

    def __len__(self):
        return len(self.input_texts)

    def MyCollate(self, batch):
        texts_list = [b[0] for b in batch]  # texts_list = ['text1', 'text2',...]
        answer_list = [b[1] for b in batch]  # targets_list = ['answer1', 'answer2', ...]

        inputs_tokenize = self.tokenizer(
            text=texts_list,
            text_target=answer_list,
            padding=True,
            truncation=True,
            max_length=300,
            return_tensors='pt'
        )
        return inputs_tokenize

            

