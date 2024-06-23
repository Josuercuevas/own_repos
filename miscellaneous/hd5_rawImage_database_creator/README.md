# HD5 dataset creation

This module will create Hd5 files with the raw pixel values coming from RGB images which have to be decoded first. The images have to be separated in different folder, for example "0/, 1/, 2/" denoting the classes.

This modules starts by scanning the "source" folder (use "dataset/" subfolder provided here) and determining how many classes we have present, afterwards each folder is scanned to determine how many samples per class are available. The samples are separated into 3 groups "Training", "Testing", and "Validation" which can be later used for training any kind of model which accepts raw RGB data as input.

The files are saved into a "destination" folder predefined by the user, use the subfolder provided in this project under the name of "augmented"

### CONTRIBUTORS

Josue R. Cuevas

josuercuevas@gmail.com
