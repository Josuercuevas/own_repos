from configs.conf import (LOGE, LOGW, LOGI)
import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        LOGI("Preparing DataModuleFromConfig ...")
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            LOGI("Setting Training dataset as it is present in the yaml config ...")
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
            LOGI(f"Training dataloader has been set to: {self.train_dataloader}")
        if validation is not None:
            LOGI("Setting Validation dataset as it is present in the yaml config ...")
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
            LOGI(f"Validation dataloader has been set to: {self.val_dataloader}")
        if test is not None:
            LOGI("Setting Test dataset as it is present in the yaml config ...")
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
            LOGI(f"Test dataloader has been set to: {self.test_dataloader}")
        if predict is not None:
            LOGI("Setting Prediction dataset as it is present in the yaml config ...")
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
            LOGI(f"Predict dataloader has been set to: {self.predict_dataloader}")
        self.wrap = wrap
        LOGI("DataModuleFromConfig ready!")

    # here is where the dataloader are configured
    def prepare_data(self):
        LOGI("Preparing Dataloader ...")
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)
        LOGI("Dataloader ready!")

    # here is where the dataloader are mapped for training/testing/validation/prediction
    def setup(self, stage=None):
        LOGI("Setting up Dataloader ...")
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])
        LOGI(f"Dataloader set!")

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            LOGI("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            LOGI("Project config")
            LOGI(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            LOGI("Lightning config")
            LOGI(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                LOGE(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    # see https://github.com/williamFalcon/pytorch-lightning-vae/issues/7
    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class TrainerEngine:
    def __init__(self, arg_parser):
        # forldername to be used for the logs
        self.now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        sys.path.append(os.getcwd())

        self.parser = arg_parser
        self.parser = Trainer.add_argparse_args(self.parser)
        self.opt, self.unknown = self.parser.parse_known_args()

        if self.opt.name and self.opt.resume:
            LOGE("-n/--name and -r/--resume cannot be specified both."
                "If you want to resume training in a new log folder, "
                "use -n/--name in combination with --resume_from_checkpoint")
            raise ValueError
        
        if self.opt.resume:
            if not os.path.exists(self.opt.resume):
                LOGE(f"Cannot find {self.opt.resume}")
                raise ValueError
            if os.path.isfile(self.opt.resume):
                LOGI(f"Loading modelfile from: {self.opt.resume}")
                paths = self.opt.resume.split("/")
                self.logdir = "/".join(paths[:-2])
                ckpt = self.opt.resume
            else:
                LOGI(f"Loading modelfile from folderpath: {self.opt.resume}")
                assert os.path.isdir(self.opt.resume), self.opt.resume
                self.logdir = self.opt.resume.rstrip("/")
                ckpt = os.path.join(self.logdir, "checkpoints", "last.ckpt")

            self.opt.resume_from_checkpoint = ckpt
            base_configs = sorted(glob.glob(os.path.join(self.logdir, "configs/*.yaml")))
            self.opt.base = base_configs + self.opt.base
            _tmp = self.logdir.split("/")
            self.nowname = _tmp[-1]
        else:
            if self.opt.name:
                name = "_" + self.opt.name
            elif self.opt.base:
                cfg_fname = os.path.split(self.opt.base[0])[-1]
                cfg_name = os.path.splitext(cfg_fname)[0]
                name = "_" + cfg_name
            else:
                name = ""
            
            self.nowname = self.now + name + self.opt.postfix
            self.logdir = os.path.join(self.opt.logdir, self.nowname)

        self.ckptdir = os.path.join(self.logdir, "checkpoints")
        self.cfgdir = os.path.join(self.logdir, "configs")
        seed_everything(self.opt.seed)

        LOGW(f"Configuration is:")
        LOGW(f"opt:\n{self.opt}")
        LOGW(f"ckptdir:\n{self.ckptdir}")
        LOGW(f"cfgdir:\n{self.cfgdir}")
        LOGW(f"logdir:\n{self.logdir}")
        LOGW(f"nowname:\n{self.nowname}")

    def run(self):
        try:
            # init and save configs
            configs = [OmegaConf.load(cfg) for cfg in self.opt.base]
            cli = OmegaConf.from_dotlist(self.unknown)
            config = OmegaConf.merge(*configs, cli)
            lightning_config = config.pop("lightning", OmegaConf.create())
            # merge trainer cli with config
            trainer_config = lightning_config.get("trainer", OmegaConf.create())
            # default to ddp
            trainer_config["strategy"] = "ddp"
            # to avoid warning "PossibleUserWarning: `max_epochs` was not set"
            trainer_config["max_epochs"] = -1
            for k in nondefault_trainer_args(self.opt):
                trainer_config[k] = getattr(self.opt, k)
            if not "gpus" in trainer_config:
                LOGW(f"Running purely on CPU")
                del trainer_config["strategy"]
                cpu = True
            else:
                gpuinfo = trainer_config["gpus"]
                LOGW(f"Running on GPUs {gpuinfo}")
                cpu = False
            trainer_opt = argparse.Namespace(**trainer_config)
            lightning_config.trainer = trainer_config

            LOGI(f"configs:\n{configs}")
            LOGI(f"lightning_config:\n{lightning_config}")
            LOGI(f"trainer_config:\n{trainer_config}")
            LOGI(f"trainer_opt:\n{trainer_opt}")

            # model
            model = instantiate_from_config(config.model)
            LOGI(f"Model:\n{model}")

            # trainer and callbacks
            trainer_kwargs = dict()

            # default logger configs
            default_logger_cfgs = {
                "testtube": {
                    "target": "pytorch_lightning.loggers.TensorBoardLogger",
                    "params": {
                        "name": "testtube",
                        "save_dir": self.logdir,
                    }
                },
            }
            default_logger_cfg = default_logger_cfgs["testtube"]
            if "logger" in lightning_config:
                logger_cfg = lightning_config.logger
            else:
                logger_cfg = OmegaConf.create()
            logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
            trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

            LOGI(f"default_logger_cfgs:\n{default_logger_cfgs}")
            LOGI(f"logger_cfg:\n{logger_cfg}")
            LOGI(f"logger_cfg:\n{logger_cfg}")

            # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
            # specify which metric is used to determine best models
            default_modelckpt_cfg = {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": self.ckptdir,
                    "filename": "{epoch:06}",
                    "verbose": True,
                    "save_last": True,
                }
            }
            if hasattr(model, "monitor"):
                LOGI(f"Monitoring {model.monitor} as checkpoint metric.")
                default_modelckpt_cfg["params"]["monitor"] = model.monitor
                default_modelckpt_cfg["params"]["save_top_k"] = 3

            if "modelcheckpoint" in lightning_config:
                modelckpt_cfg = lightning_config.modelcheckpoint
            else:
                modelckpt_cfg =  OmegaConf.create()
            modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
            LOGI(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
            
            if version.parse(pl.__version__) < version.parse('1.4.0'):
                trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)
            
            # add callback which sets up log directory
            default_callbacks_cfg = {
                # setting up everything for training and testing
                "setup_callback": {
                    "target": "trainer.trainer_main.SetupCallback",
                    "params": {
                        "resume": self.opt.resume,
                        "now": self.now,
                        "logdir": self.logdir,
                        "ckptdir": self.ckptdir,
                        "cfgdir": self.cfgdir,
                        "config": config,
                        "lightning_config": lightning_config,
                    }
                },
                # in charge of logging some images for visualization purposes
                "image_logger": {
                    "target": "trainer.trainer_main.ImageLogger",
                    "params": {
                        "batch_frequency": 750,
                        "max_images": 4,
                        "clamp": True
                    }
                },
                # Automatically monitor and logs learning rate for learning rate schedulers during training.
                "learning_rate_logger": {
                    "target": "trainer.trainer_main.LearningRateMonitor",
                    "params": {
                        "logging_interval": "step",
                        # "log_momentum": True
                    }
                },
                # For tracking purposes or resetting things up if training starts
                "cuda_callback": {
                    "target": "trainer.trainer_main.CUDACallback"
                },
            }
            if version.parse(pl.__version__) >= version.parse("1.4.0"):
                default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

            if "callbacks" in lightning_config:
                callbacks_cfg = lightning_config.callbacks
            else:
                callbacks_cfg = OmegaConf.create()

            if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:
                LOGI(f"Caution: Saving checkpoints every n train steps without deleting. This might require some free space.")
                default_metrics_over_trainsteps_ckpt_dict = {
                    "metrics_over_trainsteps_checkpoint":
                        {"target": "pytorch_lightning.callbacks.ModelCheckpoint",
                        "params": {
                            "dirpath": os.path.join(self.ckptdir, "trainstep_checkpoints"),
                            "filename": "{epoch:06}-{step:09}",
                            "verbose": True,
                            "save_top_k": -1,
                            "every_n_train_steps": 10000,
                            "save_weights_only": True
                        }
                        }
                }
                default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

            callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
            if "ignore_keys_callback" in callbacks_cfg and hasattr(trainer_opt, "resume_from_checkpoint"):
                callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
            elif "ignore_keys_callback" in callbacks_cfg:
                del callbacks_cfg['ignore_keys_callback']

            trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

            LOGI(f"trainer_kwargs:\n{trainer_kwargs}")

            LOGI("#### Trainer has the following accelarators #####")
            trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
            trainer.logdir = self.logdir

            # data which is configured in the yaml file you entered in your cli
            data = instantiate_from_config(config.data)
            # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
            # calling these ourselves should not be necessary but it is.
            # lightning still takes care of proper multiprocessing though
            data.prepare_data()
            data.setup()
            LOGI("#### Data #####")
            for k in data.datasets:
                LOGI(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

            # configure learning rate
            bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
            if not cpu:
                ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
            else:
                ngpu = 1
            if 'accumulate_grad_batches' in lightning_config.trainer:
                accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
            else:
                accumulate_grad_batches = 1
            LOGI(f"accumulate_grad_batches = {accumulate_grad_batches}")
            lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
            if self.opt.scale_lr:
                model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
                LOGI(
                    f"Setting learning rate to {model.learning_rate:.2e} = {accumulate_grad_batches}"
                    f" (accumulate_grad_batches) * {ngpu} (num_gpus) * {bs} (batchsize) *"
                    f" {base_lr:.2e} (base_lr)")
            else:
                model.learning_rate = base_lr
                LOGI("++++ NOT USING LR SCALING ++++")
                LOGI(f"Setting learning rate to {model.learning_rate:.2e}")

            # allow checkpointing via USR1 in case kill signal is sent
            def melk(*args, **kwargs):
                # run all checkpoint hooks
                if trainer.global_rank == 0:
                    LOGW("Summoning checkpoint.")
                    ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
                    trainer.save_checkpoint(ckpt_path)


            # in case we need to debug what has happened
            def divein(*args, **kwargs):
                if trainer.global_rank == 0:
                    import pudb;
                    pudb.set_trace()


            import signal

            signal.signal(signal.SIGUSR1, melk)
            signal.signal(signal.SIGUSR2, divein)

            # this is where the magic happens and where the training process starts
            if self.opt.operation == "train":
                try:
                    # reference https://lightning.ai/docs/pytorch/stable/common/trainer.html#fit
                    trainer.fit(model, data)
                except Exception:
                    melk() # make sure we save things if we need to quit for any reason
                    raise
            if not self.opt.no_test and not trainer.interrupted:
                # reference https://lightning.ai/docs/pytorch/stable/common/trainer.html#test
                trainer.test(model, data)
        except Exception as e:
            LOGE(f"Got a runtime error: {str(e)}")
            if trainer.global_rank == 0:
                try:
                    import pudb as debugger
                except ImportError:
                    import pdb as debugger
                debugger.post_mortem()
            raise
        finally:
            # move newly created debug project to debug_runs
            if not self.opt.resume and trainer.global_rank == 0:
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "debug_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                os.rename(self.logdir, dst)
            if trainer.global_rank == 0:
                LOGI(trainer.profiler.summary())
