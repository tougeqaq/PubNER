import os
import torch 
import logging
import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, BartForConditionalGeneration, get_constant_schedule_with_warmup

import config
import utils
from trainer import Trainer
from dataloader import MyDataset
from model_updater import ModelUpdater



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,  default='./config/only_biodatas.json')
    parser.add_argument('--stage', type=int,  default=1)
    parser.add_argument('--model', type=str, default='BioBART-large')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epoch_num', type=int)
    parser.add_argument('--max_lr', type=float)
    parser.add_argument('--warm_up_rate', type=float)
    parser.add_argument('--get_data_scale', type=float)
    args = parser.parse_args()
    config = config.Config(args)

    # 加载tensorboard
    writer = utils.get_tensorboard(config.save_tenserboard_folder, config.dataset_name, config.model_name, config.max_lr)

    # 指定GPU
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{config.device}')

    # 加载model和tokenizer
    model_path = os.path.join(config.model_folder, config.model_name)
    special_tokens = ['<entity_start>', '<entity_end>']
    VatorUpdater = ModelUpdater(model_path, special_tokens)
    model, tokenizer = VatorUpdater.update_model_and_tokenizer()
    # 如果是阶段1的训练(或直接训练), 即--stage = 1,则不加载其他权重；反之则加载
    if config.stage == 1:
        model = model.to(device)
    elif config.stage == 2:
        # 注意修改需要加载的权重的路径
        state_dict = torch.load("/home/sda/wangzhijun/MyLearningCode/PubtatorNewTrain/checkpoints/BioBART-large/7_pubtator_6w_5e-06_best_model.pt")
        model.load_state_dict(state_dict)
        model = model.to(device)
        config.dataset_name = config.dataset_name + '_with_pubtator_6w'
        print(f"--dataset name is {config.dataset_name}")
        
    config.model = model
    config.tokenizer = tokenizer
    
    # 指定logger
    logger = utils.get_logger(config.dataset_name, config.model_name, config.save_log_folder)
    logger.info(config)
    config.logger = logger

    # 加载数据
    logger.info("--Loading Data--")
    trainset = MyDataset(config, is_train_set=True)
    trainloader = DataLoader(
        trainset,
        shuffle=True,
        batch_size=config.batch_size,
        collate_fn=trainset.MyCollate
    )
    config.num_batchs = len(trainloader)
    testLoaders = {}
    for test_file in os.listdir(config.testset_folder):
        test_dataset_path = os.path.join(config.testset_folder, test_file)
        config.testset_path = test_dataset_path
        testset = MyDataset(config, is_train_set=False)
        key_name = test_file.split('-test')[0]
        testLoaders[key_name] = DataLoader(
            testset, 
            shuffle=False, 
            batch_size=config.batch_size,
            collate_fn=testset.MyCollate
        )

    MyTrainer = Trainer(config)

    best_f1 = 0
    best_epoch = 0
    logger.info(f"--Start Training with {config.max_lr}--")
    for epoch in range(config.epoch_num):
        train_loss = MyTrainer.train(config, trainloader)
        logger.info(f"Epoch {epoch+1}/{config.epoch_num}, Train Loss: {train_loss: .4f}")

        writer.add_scalar(f'Train Loss/{config.model_name}', train_loss, epoch+1)
        writer.add_scalar(f'lr/{config.model_name}', MyTrainer.scheduler.get_lr()[0], epoch+1)


        sum_f1 = 0
        sum_P = 0
        sum_R = 0
        test_num = 0

        if epoch > 5:
            for testfile_name, testfile_dataloader in testLoaders.items():
                # if testfile_name == config.dataset_name or config.dataset_name == 'all_data' or config.dataset_name[0:7] == 'pubtator':
                P, R, f1_score, pre_and_true = MyTrainer.eval(config, testfile_dataloader)
                logger.info(f"--{testfile_name} f1-score: {f1_score: .4f}, P: {P:.4f}, R: {R:.4f}")
                writer.add_scalar(f'{testfile_name} F1-score/{config.model_name}', f1_score, epoch+1)
                utils.save_generate_answer(config.save_predict_folder, testfile_name, config.model_name, pre_and_true, epoch, f1_score, config.max_lr)
                sum_f1 += f1_score
                sum_P += P
                sum_R += R
                test_num += 1
        
            # 记录最大的f1值，并记录最优权重
            if sum_f1 > best_f1:
                best_f1 = sum_f1
                best_epoch = epoch + 1
                best_model_state_dict = model.state_dict().copy()
            logger.info(f"------------------The aver f1 score: {sum_f1/test_num: .4f} || the best f1 score: {best_f1/test_num: .4f}, best epoch is {best_epoch}")
            writer.add_scalar(f'aver F1-score/{config.model_name}', sum_f1/test_num, epoch+1)
            writer.add_scalar(f'aver Precision/{config.model_name}', sum_P/test_num, epoch+1)
            writer.add_scalar(f'aver Recall/{config.model_name}', sum_R/test_num, epoch+1)
            print(f"Epoch {epoch+1}/{config.epoch_num}, Train Loss: {train_loss: .4f}")
        
        
    if best_model_state_dict is not None:
        save_path = utils.get_specified_path(config.save_checkpoint_folder, f"{best_epoch}_{config.dataset_name}_{config.max_lr}_best_model", '.pt', config.model_name)
        torch.save(best_model_state_dict, save_path)
    logger.info(f"The max f1 score: {best_f1/test_num: .4f}, the best epoch is {best_epoch}")
    writer.close()

        





    

