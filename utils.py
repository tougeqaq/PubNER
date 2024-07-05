import logging
import pickle
import pytz
from datetime import datetime
import time
import os
from torch.utils.tensorboard import SummaryWriter

# 创建本地时间用于记录Log
def get_local_time():
    local_tz = pytz.timezone('Asia/Shanghai')
    local_time = datetime.now(local_tz)
    formatted_time = local_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time

# 设置Log中的时间戳为本地时间 
class CustomFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt, datefmt)
        self.tz = tz

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

# 创建不存在的文件夹: rootdir/subdirectory_1/subdirectory_2/.../(file_name+suffix)
# (log_folder, log_file_name, '.log', model_name) --> log_folder/model_name/log_file_name.log
def get_specified_path(rootdir, file_name, suffix, *subdirectories):
    subdirectory_path = rootdir
    for subdir in subdirectories:
        subdirectory_path = os.path.join(subdirectory_path, subdir)
        if not os.path.exists(subdirectory_path):
            os.makedirs(subdirectory_path)
    if file_name != '':
        file_name = file_name + suffix
        file_path = os.path.join(subdirectory_path, file_name)
    else:
        file_path = subdirectory_path
    
    return file_path

# 创建Logger
def get_logger(dataset_name, model_name, log_folder):
    formatted_time = get_local_time().replace(':','-').replace(' ','_')
    log_file_name = f"{dataset_name}_{formatted_time}"
    pathname = get_specified_path(log_folder, log_file_name, '.log', model_name)
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)
    
    ## 定义日志消息格式和日期格式
    formatter = CustomFormatter("%(asctime)s - %(levelname)s: %(message)s",
                                datefmt = '%Y-%m-%d %H:%M:%S',
                                tz=pytz.timezone('Asia/Shanghai')
                                )

    # 创建文件处理器，将日志消息写入文件
    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.INFO) ## 其实这里不用设置，因为根日志记录器设置了Level，它是全局的过滤器
    file_handler.setFormatter(formatter)

    # 创建流处理器，将日志消息输出到控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # 将两个处理器都添加到日志记录器种
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

# 加载tensorboard
def get_tensorboard(tensorboard_folder, datasetname, modelname, max_lr):
    tensorboard_path = get_specified_path(tensorboard_folder, datasetname, '' ,modelname)
    writer = SummaryWriter(
        log_dir = tensorboard_path, 
        filename_suffix=f"--{max_lr}" # 给tensorboard文件加后缀，以区分
    )
    return writer

# 抽取生成语句中的实体和实体span
def extract_entities_with_spans(sentence):
    # <entity_start>长度为14, <entity_end>长度为12
    s_flag = 0 ## <entity_start>位置
    e_flag = 0 ## <entity_end>位置

    s_num = 0 ## <entity_start>个数
    e_num = 0 ## <entity_end>个数

    entity_info = []
    s_index = 0
    e_index = 0

    nested_entity = 0

    while s_index < len(sentence):
        if sentence[s_index: s_index+14] == '<entity_start>':
            nested_entity = 0
            s_flag = s_index
            e_index = s_index + 1
            while e_index < len(sentence) and sentence[e_index: e_index+14] != '<entity_start>':
                if sentence[e_index: e_index+12] == '<entity_end>':
                    nested_entity += 1

                    entity = sentence[s_index+14 : e_index]
                    entity = entity.replace('<entity_start>', '').replace('<entity_end>', '')

                    if nested_entity > 1:
                        entity_start = s_flag
                    else:
                        entity_start = s_flag - 14*s_num - 12*e_num
                    entity_span = [entity_start, entity_start+len(entity)]
                    entity_info.append(
                        {
                            'entity': entity,
                            'span': entity_span
                        }
                    )
                    e_num += 1
                e_index += 1
            s_num += 1
        s_index += 1


    return entity_info

# 保存生成的语句，用于对比效果
def save_generate_answer(save_path, filename, modelname, true_pre_sentence, epoch, f1_score, max_lr):

    save_output_path = get_specified_path(save_path, filename, f'_{max_lr}.txt', modelname)
    with open(save_output_path, 'a+', encoding='utf-8') as f:
        predict_labels = true_pre_sentence[0]
        true_labels = true_pre_sentence[1]
        f.write(f'Epoch: {epoch+1}'+ '\n')
        for pre, true in zip(predict_labels, true_labels):
            f.write('pred: ' + str(pre) + '\n')
            f.write('true: ' + str(true) + '\n')
            f.write('当前epoch的F值:'+str(f1_score))
            f.write('-----------------------' + '\n')