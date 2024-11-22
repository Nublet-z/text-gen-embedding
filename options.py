import torch

class Options():
    def __init__(self):
        self.model_name_or_path = 'patrickvonplaten/bert2bert_cnn_daily_mail'
        self.batch_size = 16
        self.num_epochs = 10
        self.hidden_size = 768
        self.embedding_dim = 768
        self.seq_length = 80
        self.print_every = 16
        self.is_cuda = torch.cuda.is_available()
        self.emb = "bert"
        self.pad_token_id = 0
        self.vocab_size = 30522
        self.d_model = 768
        self.data_type = "hf"
        self.dataroot = "asset"
        self.isTrain = True
        self.gpu_ids = []
        self.name = "Bert_emb"
        self.lambda_identity = 0.5
        self.lambda_A = 10.0
        self.lambda_B = 10.0
        self.gamma = 0.001 # temperature

        self.input_nc = 1  # default value
        self.output_nc = 1  # default value
        self.dataroot = 'asset'  # default value
        self.f = 'asset'  # default value
        self.checkpoints_dir = './checkpoints'  # default value
        self.model = 'cycle_gan'  # default value
        self.netD = 'basic'  # default value
        self.netG = 'lstmg2'  # default value
        self.n_layers_G = 1
        self.n_layers_D = 2  # default value
        self.norm = 'instance'  # default value
        self.init_type = 'normal'  # default value
        self.init_gain = 0.02  # default value
        self.no_dropout = False  # default value (action='store_true' means False by default)
        self.pooling_strategy = 'max'  # default value
        self.dataset_mode = 'unaligned'  # default value
        self.data_is_preprocessed = False  # default value
        self.direction = 'AtoB'  # default value
        self.serial_batches = True  # default value (action='store_true')
        self.num_threads = 2  # default value
        self.max_dataset_size = float("inf")  # default value
        self.preprocess = 'resize_and_crop'  # default value
        self.no_flip = False  # default value (action='store_true' means False by default)
        self.epoch = 'latest'  # default value
        self.load_iter = 0  # default value
        self.verbose = False  # default value (action='store_true' means False by default)
        self.use_wandb = False  # default value (action='store_true' means False by default)
        self.wandb_project_name = 'CycleGAN-and-pix2pix'  # default value

        self.display_ncols = 4
        self.display_id = 0
        self.display_server = "http://localhost"
        self.display_env = 'main'
        self.display_port = 8097
        self.print_freq = 16
        self.no_html = True
        self.save_latest_freq = 5000
        self.save_epoch_freq = 5
        self.save_by_iter = False  # action='store_true' means False by default
        self.continue_train = False  # action='store_true' means False by default
        self.epoch_count = 1
        self.phase = 'train'
        self.n_epochs = 50
        self.n_epochs_decay = 50
        self.beta1 = 0.5
        self.lr = 1e-4
        self.gan_mode = 'lsgan'
        self.pool_size = 50
        self.lr_policy = 'linear'
        self.lr_decay_iters = 50
        self.clip = 5

        # Initialize argument parsing
        self.initialized = True