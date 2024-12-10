from models.autoencoder import AutoEncoderUnet
from utils.configs import (LOGE, LOGW, LOGI, 
                           ACTIVATION_FUNCTIONS)
import torch
from torchviz import make_dot

class TestAutoEncoderUnet:
    """
    Unit test to determine if Unet based autoencoder can be created
    successfully and some dummy data are forward passed through the model
    """
    def __init__(self, dst_path):
        self.img_channels=3
        self.initial_channels=128
        self.channel_mults=(1, 2, 2, 2)
        self.num_res_blocks = 2
        self.time_emb_dim=128 * 4
        self.time_emb_scale = 1
        self.norm="group_norm"
        self.dropout=0.1
        self.activation="silu"
        self.attention_res=(1,)
        self.num_classes=10
        self.num_groups = 32
        self.image_pad=0
        self.num_timesteps = 1000
        self.dst_path = dst_path
        self.success = False
        self.activations = ACTIVATION_FUNCTIONS

        self.__run()
    
    def __run(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        dummy_input = torch.randn((1,3,64,64), device=device)
        LOGI(dummy_input)
        tstep = torch.randint(1, self.num_timesteps, (1,), device=device)
        LOGW(tstep)
        yclass = torch.randint(1, self.num_classes, (1,), device=device)
        LOGI(yclass)

        LOGW("Creating Unet Model")
        try:
            model = AutoEncoderUnet(
                img_channels=self.img_channels,
                initial_channels=self.initial_channels,
                channel_mults=self.channel_mults,
                num_res_blocks=self.num_res_blocks,
                time_emb_dim=self.time_emb_dim,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activations[self.activation],
                attention_res=self.attention_res,
                num_classes=self.num_classes,
                num_groups=self.num_groups,
                image_pad=self.image_pad
            )
        except Exception as e:
            LOGE(f"Test Unet Model creation did not pass, error: {str(e)}")
        
        LOGI(f"The model looks like this: {model}")

        LOGW("Forward Pass")
        try:
            yhat = model(x=dummy_input, tstep_emb=tstep, yclass=yclass)
        except Exception as e:
            LOGE(f"Test Unet Model inference did not pass, error: {str(e)}")

        LOGW("Creating image")
        make_dot(yhat, params=dict(list(model.named_parameters()))).render(f"{self.dst_path}/doc/unet_visualizer.dot", format="png")

        self.success = True