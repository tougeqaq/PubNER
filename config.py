import json

class Config:
    def __init__(self, args):
        with open(args.config, 'r', encoding="utf-8") as f:
            config = json.load(f)

        self.dataset_name = config["dataset_name"]
        self.model_name = config["model_name"]
        self.trainset_path = config["trainset_path"]
        self.testset_folder = config["testset_folder"]
        self.model_folder = config["model_folder"]
        self.save_checkpoint_folder = config["save_checkpoint_folder"]
        self.save_log_folder = config["save_log_folder"]
        self.save_predict_folder = config["save_predict_folder"]
        self.save_tenserboard_folder = config["save_tenserboard_folder"]

        self.batch_size = config["batch_size"]
        self.batch_grad_accumlate = config["batch_grad_accumlate"]
        self.epoch_num = config["epoch_num"]
        self.max_lr = config["max_lr"]
        self.warm_up_rate = config["warm_up_rate"]
        self.get_data_scale = config["get_data_scale"] ## 取数据比例
        self.generate_hyper = config["generate_hyper"]

        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())