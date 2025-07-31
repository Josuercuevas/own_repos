# SVO2 - Data Extraction

This project extracts IMU data and left and right view frames from SVO2 video files from Stereolabs.

## Files
| File          | Description                                                                                       |
|---------------|---------------------------------------------------------------------------------------------------|
| `run_svo2data.py` | This script is the entry point for processing SVO2 files. It initializes the `SVO2Data` class and starts the processing. |
| `svo2data.py`     | This module contains the `SVO2Data` class, which handles the extraction of IMU data and left and right view frames from SVO2 files. |
| `Dockerfile`      | The Dockerfile defines the environment for running the SVO2Data extraction. It sets up the necessary dependencies and configurations. |
| `Makefile`        | The Makefile provides commands to build the Docker image and run the container.                                                   |

## Building and launching Docker

To build the Docker image and run the container, use the following commands:

1. Build the Docker image:
    ```bash
    make build
    ```

2. Run the Docker container:
    ```bash
    docker run --rm -v $(pwd):/workspace -w /workspace prime-workspace
    ```
    In case we are using **VSCode**, we can use the ***.devcontainer*** file in this folder using :
    <kbd>⌘ Command</kbd> + <kbd>⇧ Shift</kbd> + <kbd>P</kbd> in OSX to open folder in your container. To use this
    you  must install [Dev Containers plugin](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).


## Usage

### Running the SVO2Data Extraction

To run the SVO2Data extraction, use the following command:

```bash
python run_svo2data.py -i <path_to_svo2_file_or_directory>
```

Replace `<path_to_svo2_file_or_directory>` with the path to your SVO2 file or directory containing SVO2 files.

## Output
This program creates a separate folder for each detected SVO2. Within each folder, there will be two MP4 videos—one 
for the left view and one for the right view—along with a CSV file that contains the IMU values for each processed frame.
### IMU data order 
The IMU data is ordered by frame ID in a csv file, inside each file we have one row per frame, the orders is:
```
frame-id, accelerometer-x, accelerometer-y, accelerometer-z, gyroscope-0, gyroscope-1, gyroscope-2
```