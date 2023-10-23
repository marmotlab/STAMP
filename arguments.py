import math


class Arguments:
    def __init__(self):
        self.use_gpu_runner = False
        self.use_gpu_driver = True
        self.cuda_devices = [0]
        self.episode_steps = 256
        self.num_meta = 16
        self.num_minibatch = 16
        self.buffer_size = int(self.num_meta * self.episode_steps)
        self.minibatch_size = int(self.buffer_size // self.num_minibatch)
        self.update_epochs = 4
        self.curriculum = True  # curriculum learning

        self.lr = 1e-4
        self.lr_decay_step = 64
        self.gamma = 0.99
        self.gae_lambda = 0

        self.embedding_dim = 128
        self.high_info_thre = math.exp(-0.5)  # defined target area
        self.adaptive_kernel = False
        self.budget_size = (39.9999, 40)  # monitoring horizon
        self.graph_size = (100, 201)   # graph size - randomized during training
        self.history_size = (50, 101)  # history sequence length
        self.k_size = 10  # knn - number of neighboring nodes
        self.target_size = (2, 6)
        self.history_stride = 5  # set 1 to disable pooling
        self.prior_measurement = True  # True for peak measures

        self.summary_window = 1
        self.run_name = 'run'
        self.model_path = f'models/{self.run_name}'
        self.train_path = f'runs/{self.run_name}'
        self.gifs_path = f'gifs/{self.run_name}'
        self.load_model = False
        self.use_wandb = False
        if self.use_wandb:
            self.project_name = 'STAMP'
            self.wandb_notes = ''
            self.wandb_id = ''
        self.save_img_gap = 0  # 0 to turn off
        self.save_files = False


class ArgumentsEval(Arguments):
    def __init__(self):
        super().__init__()
        self.high_info_thre = 'change in arguments'
        self.prior_measurement = 'change in arguments'

        self.run_name = 'run'
        self.model_path = f'models/{self.run_name}'
        self.result_path = self.run_name
        self.cuda_devices = [0]
        self.num_meta = 1  # number of threads
        self.num_eval = 1  # number of evaluation instances, neval % nmeta == 0

        self.budget_size = 30
        self.graph_size = 200
        self.history_size = 100  # history sequence length
        self.k_size = 10  # knn - number of neighboring nodes
        self.target_size = 6
        self.target_speed = 1/20
        self.history_stride = 5

        self.save_results = False  # save results to csv
        self.save_img_gap = 0  # 0 to turn off, >=1 to save images


arg = Arguments()
arg_eval = ArgumentsEval()
