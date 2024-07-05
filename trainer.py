import os
import torch 
import logging
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, BartForConditionalGeneration, get_cosine_schedule_with_warmup

import utils

class Trainer(object):
    def __init__(self, config):
        self.model = config.model
        # self.trainloader = trainloader
        self.num_batchs = config.num_batchs

        updatas_total = config.epoch_num * self.num_batchs // config.batch_grad_accumlate
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.max_lr)
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer = self.optimizer,
            num_warmup_steps = config.warm_up_rate * updatas_total,
            num_training_steps = updatas_total
        )

    def train(self, config, trainloader):
        self.model.train()
        total_loss = 0
        num_batchs = len(trainloader)
        with tqdm(total=num_batchs, desc='Epoch', dynamic_ncols=True) as pbar:
            for batch_ids, batch in enumerate(trainloader):
                inputs = batch
                inputs = inputs.to(config.device)
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()
                if batch_ids % config.batch_grad_accumlate == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                pbar.set_postfix({"Loss": f'{loss.item(): .4f}'})
                pbar.update(1)
                total_loss += loss.item()
        return total_loss/self.num_batchs
    
    def eval(self, config, testloader):
        self.model.eval()

        pred_result = []
        label_result = []

        true_positives = 0 # 预测正确的实体总数
        total_predictions = 0 # 预测出来的实体总数
        total_ground_truths = 0 # 真实标签中实体总数

        with torch.no_grad():
            num_batchs = len(testloader)
            with tqdm(total=num_batchs, desc="Eval", dynamic_ncols=True) as pbar:
                for batch in testloader:
                    inputs = batch
                    inputs = inputs.to(config.device)
                    labels = inputs.pop('labels') # 把答案标签去掉
                    generate_hyper = dict(
                                            num_beams=3,
                                            max_new_tokens = 512,
                                            early_stopping= False,
                                            length_penalty = 0,
                                            no_repeat_ngram_size = 0, # 防止生成模型做实体抽取任务时，抑制重复类别的生成输出
                                    )
                    outputs = self.model.generate(**inputs, **generate_hyper)
                    
                    pred_answers = config.tokenizer.batch_decode(outputs, skip_special_tokens=False)
                    true_answers = config.tokenizer.batch_decode(labels, skip_special_tokens=False)

                    pred_result += pred_answers
                    label_result += true_answers

                    for pred_sentence, true_sentence in zip(pred_answers, true_answers):
                        pred_sentence = pred_sentence.replace('</s>','').replace('<s>','').replace('<pad>', '').replace('<unk>','').replace('<mask>','')
                        true_sentence = true_sentence.replace('</s>','').replace('<s>','').replace('<pad>', '').replace('<unk>','').replace('<mask>','')

                        ## [{'entity': 'B . germanica cyclophilin', 'span': [4, 29]}, {'entity': 'B . germanica cyclophilin amino acid sequence', 'span': [4, 49]}]
                        predict_label =  utils.extract_entities_with_spans(pred_sentence)
                        true_label = utils.extract_entities_with_spans(true_sentence)
                        ## 转换为集合以便于比较
                        ground_truth_set = {(gt['entity'], tuple(gt['span'])) for gt in true_label}
                        prediction_set = {(pred['entity'], tuple(pred['span'])) for pred in predict_label}

                        ## 计算正确预测的实体数
                        true_positives += len(ground_truth_set & prediction_set)
                        
                        ## 计算预测的实体总数
                        total_predictions += len(prediction_set)
                        
                        ## 计算测试集中的实体总数
                        total_ground_truths += len(ground_truth_set)
                        
                    pbar.update(1)   
                    ## 计算精确率
                    precision = true_positives / total_predictions if total_predictions > 0 else 0
                    
                    ## 计算召回率
                    recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0
                    if precision + recall == 0:
                        f1_score = 0
                    else:
                        f1_score = 2 * (precision * recall) / (precision + recall)
                    
        return precision, recall, f1_score, (true_answers,pred_answers)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)





