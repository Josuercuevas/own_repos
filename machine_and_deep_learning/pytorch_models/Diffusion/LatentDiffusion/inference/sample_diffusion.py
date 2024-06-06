from configs.conf import (LOGE, LOGW, LOGI)
import os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange
from omegaconf import OmegaConf
from PIL import Image
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


class SampleDiffusionUnconditional:
    def __init__(self, arg_parser):
        self.now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        sys.path.append(os.getcwd())
        self.command = " ".join(sys.argv)
        self.parser = arg_parser
        self.opt, self.unknown = self.parser.parse_known_args()
        self.ckpt = None
        self.imglogdir = None
        self.numpylogdir = None
        self.rescale = lambda x: (x + 1.) / 2.

        # getting model checkpoint
        if not os.path.exists(self.opt.resume):
            LOGE(f"Cannot find checkpoint at {self.opt.resume}")
            raise ValueError(f"Cannot find checkpoint at {self.opt.resume}")
        if os.path.isfile(self.opt.resume):
            # paths = opt.resume.split("/")
            try:
                self.logdir = '/'.join(self.opt.resume.split('/')[:-1])
                # idx = len(paths)-paths[::-1].index("logs")+1
                LOGI(f"Logdir is {self.logdir}")
            except ValueError:
                paths = self.opt.resume.split("/")
                idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
                self.logdir = "/".join(paths[:idx])
            ckpt = self.opt.resume
        else:
            assert os.path.isdir(self.opt.resume), f"{self.opt.resume} is not a directory"
            self.logdir = self.opt.resume.rstrip("/")
            ckpt = os.path.join(self.logdir, "model.ckpt")
        
        LOGI("constructing configs and options")
        self.base_configs = sorted(glob.glob(os.path.join(self.logdir, "config.yaml")))
        self.opt.base = self.base_configs

        LOGI(f"Loading configs with base: {self.opt.base}")
        self.configs = [OmegaConf.load(cfg) for cfg in self.opt.base]
        self.cli = OmegaConf.from_dotlist(self.unknown)
        self.config = OmegaConf.merge(*self.configs, self.cli)

        LOGI("Setting evaluation mode and device")
        self.gpu = True
        self.eval_mode = True

        if self.opt.logdir:
            self.locallog = self.logdir.split(os.sep)[-1]
            if self.locallog == "":
                self.locallog = self.logdir.split(os.sep)[-2]
            
            LOGI(f"Switching logdir from {self.logdir} to {os.path.join(self.opt.logdir, self.locallog)}")
            self.logdir = os.path.join(self.opt.logdir, self.locallog)

        LOGI(f"Loading model from {ckpt}")
        self.model, self.global_step = self.__load_model(self.config, ckpt, self.gpu, self.eval_mode)

        LOGW(f"Initialization completed with he following configuration:\n{self.config}")
    
    def get_samples(self):
        LOGI(f"global step: {self.global_step}")
        LOGI(75 * "=")
        LOGI("logging to:")
        self.logdir = os.path.join(self.logdir, "samples", f"{self.global_step:08}", self.now)
        self.imglogdir = os.path.join(self.logdir, "img")
        self.numpylogdir = os.path.join(self.logdir, "numpy")

        os.makedirs(self.imglogdir)
        os.makedirs(self.numpylogdir)
        LOGI(self.logdir)
        LOGI(75 * "=")

        # write config out
        sampling_file = os.path.join(self.logdir, "sampling_config.yaml")
        sampling_conf = vars(self.opt)

        with open(sampling_file, 'w') as f:
            yaml.dump(sampling_conf, f, default_flow_style=False)
        LOGI(sampling_conf)

        self.__run(model=self.model, logdir=self.imglogdir, eta=self.opt.eta, vanilla=self.opt.vanilla_sample,
                   n_samples=self.opt.n_samples, custom_steps=self.opt.custom_steps, batch_size=self.opt.batch_size,
                   nplog=self.numpylogdir)
    
    def __run(self, model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None):
        if vanilla:
            LOGI(f"Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.")
        else:
            LOGI(f"Using DDIM sampling with {custom_steps} sampling steps and eta={eta}")

        n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
        # path = logdir
        if model.cond_stage_model is None:
            all_images = []

            LOGI(f"Running unconditional sampling for {n_samples} samples")
            for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
                logs = self.__make_convolutional_sample(model, batch_size=batch_size, vanilla=vanilla,
                                                        custom_steps=custom_steps, eta=eta)
                n_saved = self.__save_logs(logs, logdir, n_saved=n_saved, key="sample")
                all_images.extend([self.__custom_to_np(logs["sample"])])
                if n_saved >= n_samples:
                    LOGI(f"Finish after generating {n_saved} samples")
                    break
            all_img = np.concatenate(all_images, axis=0)
            all_img = all_img[:n_samples]
            shape_str = "x".join([str(x) for x in all_img.shape])
            nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
            np.savez(nppath, all_img)
        else:
            LOGE("Currently only sampling for unconditional models supported")
            raise NotImplementedError("Currently only sampling for unconditional models supported")
    
    def __load_model_from_config(self, config, sd):
        model = instantiate_from_config(config)
        model.load_state_dict(sd,strict=False)
        model.cuda()
        model.eval()
        return model

    def __load_model(self, config, ckpt, gpu, eval_mode):
        if ckpt:
            pl_sd = torch.load(ckpt, map_location="cpu")
            global_step = pl_sd["global_step"]
        else:
            pl_sd = {"state_dict": None}
            global_step = None
        model = self.__load_model_from_config(config.model,
                                    pl_sd["state_dict"])
        return model, global_step

    def __custom_to_pil(self, x):
        x = x.detach().cpu()
        x = torch.clamp(x, -1., 1.)
        x = (x + 1.) / 2.
        x = x.permute(1, 2, 0).numpy()
        x = (255 * x).astype(np.uint8)
        x = Image.fromarray(x)
        if not x.mode == "RGB":
            x = x.convert("RGB")
        return x

    def __custom_to_np(self, x):
        # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
        sample = x.detach().cpu()
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        return sample

    def __logs2pil(self, logs, keys=["sample"]):
        imgs = dict()
        for k in logs:
            try:
                if len(logs[k].shape) == 4:
                    img = self.__custom_to_pil(logs[k][0, ...])
                elif len(logs[k].shape) == 3:
                    img = self.__custom_to_pil(logs[k])
                else:
                    LOGE(f"Unknown format for key {k}.")
                    img = None
            except:
                img = None
            imgs[k] = img
        return imgs

    @torch.no_grad()
    def __convsample(self, model, shape, return_intermediates=True, verbose=True, make_prog_row=False):
        if not make_prog_row:
            return model.p_sample_loop(None, shape,
                                    return_intermediates=return_intermediates, verbose=verbose)
        else:
            return model.progressive_denoising(None, shape, verbose=True)

    @torch.no_grad()
    def __convsample_ddim(self, model, steps, shape, eta=1.0):
        ddim = DDIMSampler(model)
        bs = shape[0]
        shape = shape[1:]
        samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
        return samples, intermediates

    @torch.no_grad()
    def __make_convolutional_sample(self, model, batch_size, vanilla=False, custom_steps=None, eta=1.0,):
        log = dict()
        shape = [batch_size, model.model.diffusion_model.in_channels, model.model.diffusion_model.image_size,
                 model.model.diffusion_model.image_size]
        with model.ema_scope("Plotting"):
            t0 = time.time()
            if vanilla:
                sample, progrow = self.__convsample(model, shape, make_prog_row=True)
            else:
                sample, intermediates = self.__convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                               eta=eta)
            t1 = time.time()
        
        x_sample = model.decode_first_stage(sample)
        log["sample"] = x_sample
        log["time"] = t1 - t0
        log["throughput"] = sample.shape[0] / (t1 - t0)
        LOGI(f"Throughput for this batch: {log['throughput']}")
        return log

    def __save_logs(self, logs, path, n_saved=0, key="sample", np_path=None):
        for k in logs:
            if k == key:
                batch = logs[key]
                if np_path is None:
                    for x in batch:
                        img = self.__custom_to_pil(x)
                        imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                        img.save(imgpath)
                        n_saved += 1
                else:
                    npbatch = self.__custom_to_np(batch)
                    shape_str = "x".join([str(x) for x in npbatch.shape])
                    nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                    np.savez(nppath, npbatch)
                    n_saved += npbatch.shape[0]
        return n_saved
