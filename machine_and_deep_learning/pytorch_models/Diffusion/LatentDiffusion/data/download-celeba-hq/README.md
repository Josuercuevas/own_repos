# celebA-HQ-dataset-download

After you have installed all the dependecies, please just execute the following command to download the dataset:

```shell
./download_celebA-HQ.sh OUTPUT_DIR
```

**Note**: The script will take several hours to download the dataset. Additionally, you need good internet connection and lots of free storage space.

Once the dataset is downloaded, move all images with the target resolution to the subfolder **celebahq/** in the parent folder, and update the files **celebahqtrain.txt** and **celebahqvalidation.txt** with the proper filenames.

At this point you are ready to start the training of your model.