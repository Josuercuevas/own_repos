import tensorflow as tf
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']




devices_available = get_available_gpus()

for gpu_av in range(len(devices_available)):
	print(str(devices_available[gpu_av]))