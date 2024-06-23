# CIFAR10 data preparation

The two scripts shared in this folder prepare the CIFAR10 dataset for training and evaluation, you just need to run the following cmd-lines:

```sh
# uncommend the 1st and last lines if you need to
./download_data.sh

python3 preprocess_data.py
```