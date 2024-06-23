export DEBIAN_FRONTEND=noninteractive
apt update
apt install htop tmux p7zip-full


# python packages
echo ">>>>>>>>>>>>>>>>>>>> installing pandas"
pip install --upgrade --force-reinstall pandas
echo ">>>>>>>>>>>>>>>>>>>> installing torchviz"
pip install --upgrade --force-reinstall torchviz
echo ">>>>>>>>>>>>>>>>>>>> installing pytorch-lightning"
pip install --upgrade --force-reinstall pytorch-lightning==1.6.5
echo ">>>>>>>>>>>>>>>>>>>> installing torch"
pip install --upgrade --force-reinstall torch==1.13.1
echo ">>>>>>>>>>>>>>>>>>>> installing torchvision"
pip install --upgrade --force-reinstall torchvision==0.13.0
echo ">>>>>>>>>>>>>>>>>>>> installing torchmetrics"
pip install --upgrade --force-reinstall torchmetrics==0.6.0
echo ">>>>>>>>>>>>>>>>>>>> installing torchtext"
pip install --upgrade --force-reinstall torchtext==0.12.0
echo ">>>>>>>>>>>>>>>>>>>> installing omegaconf"
pip install --upgrade --force-reinstall omegaconf==2.1.1
echo ">>>>>>>>>>>>>>>>>>>> installing imageio"
#pip install --upgrade --force-reinstall imageio==2.9.0
echo ">>>>>>>>>>>>>>>>>>>> installing imageio-ffmpeg"
pip install --upgrade --force-reinstall imageio-ffmpeg==0.4.2
echo ">>>>>>>>>>>>>>>>>>>> installing streamlit"
#pip install --upgrade --force-reinstall streamlit>=0.73.1
echo ">>>>>>>>>>>>>>>>>>>> installing einops"
pip install git+https://github.com/arogozhnikov/einops.git
pip install --upgrade --force-reinstall einops==0.3.0
echo ">>>>>>>>>>>>>>>>>>>> installing torch-fidelity"
#pip install --upgrade --force-reinstall torch-fidelity==0.3.0
echo ">>>>>>>>>>>>>>>>>>>> installing transformers"
pip install git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install git+https://github.com/openai/CLIP.git@main#egg=clip
pip install --upgrade --force-reinstall transformers==4.19.2
echo ">>>>>>>>>>>>>>>>>>>> installing scann & kornia"
#pip install --upgrade --force-reinstall scann
#pip install --upgrade --force-reinstall kornia==0.6.4
echo ">>>>>>>>>>>>>>>>>>>> installing taming"
pip install taming-transformers-rom1504
echo ">>>>>>>>>>>>>>>>>>>> installing test-tube"
pip install test-tube
echo ">>>>>>>>>>>>>>>>>>>> Installing Gdrive downloader"
pip install gdown
echo ">>>>>>>>>>>>>>>>>>> Installing albumentations"
pip install albumentations