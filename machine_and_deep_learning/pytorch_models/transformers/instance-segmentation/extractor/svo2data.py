from pathlib import Path
import pyzed.sl as sl
import numpy as np
import cv2
import sys
from datetime import datetime 
from tqdm import tqdm
import platform
import math

class TimestampHandler:
	def __init__(self):
		self.t_imu = sl.Timestamp()
		self.t_baro = sl.Timestamp()
		self.t_mag = sl.Timestamp()

	##
	# check if the new timestamp is higher than the reference one, and if yes, save the current as reference
	def is_new(self, sensor):
		if (isinstance(sensor, sl.IMUData)):
			new_ = (sensor.timestamp.get_microseconds() > self.t_imu.get_microseconds())
			if new_:
				self.t_imu = sensor.timestamp
			return new_
		elif (isinstance(sensor, sl.MagnetometerData)):
			new_ = (sensor.timestamp.get_microseconds() > self.t_mag.get_microseconds())
			if new_:
				self.t_mag = sensor.timestamp
			return new_
		elif (isinstance(sensor, sl.BarometerData)):
			new_ = (sensor.timestamp.get_microseconds() > self.t_baro.get_microseconds())
			if new_:
				self.t_baro = sensor.timestamp
			return new_

##
#  Function to display sensor parameters
def printSensorParameters(sensor_parameters):
	if sensor_parameters.is_available:
		print("Sensor type: " + str(sensor_parameters.sensor_type))
		print("Max rate: "  + str(sensor_parameters.sampling_rate) + " "  + str(sl.SENSORS_UNIT.HERTZ))
		print("Range: "  + str(sensor_parameters.sensor_range) + " "  + str(sensor_parameters.sensor_unit))
		print("Resolution: " + str(sensor_parameters.resolution) + " "  + str(sensor_parameters.sensor_unit))
		if not math.isnan(sensor_parameters.noise_density):
			print("Noise Density: "  + str(sensor_parameters.noise_density) + " " + str(sensor_parameters.sensor_unit) + "/√Hz")
		if not math.isnan(sensor_parameters.random_walk):
			print("Random Walk: "  + str(sensor_parameters.random_walk) + " " + str(sensor_parameters.sensor_unit) + "/s/√Hz")
	


class SVO2Data:
	def __init__(self, _args):
		self.source = _args.source
		self._files2process = []
		if Path(self.source).exists() is False:
			raise FileNotFoundError(f'{self.source} does not exist.')
		if Path(self.source).is_file():
			self._files2process.append(self.source)
		if Path(self.source).is_dir():
			self._files2process = list(Path(self.source).glob('*.svo2'))
		self._MACHINE = platform.machine()
		self.width = None
		self.height = None
		self.fps = None

	def process(self):
		output_folder = None
		if Path(self.source).is_dir():
			output_folder = f"output/{Path(self.source).name}"
		if Path(self.source).is_file():
			output_folder = f"output/{Path(self.source).stem}"
		Path(output_folder).mkdir(parents=True, exist_ok=True)
		init_parameters= sl.InitParameters(
			depth_mode=sl.DEPTH_MODE.NEURAL,
			coordinate_units=sl.UNIT.METER,
			coordinate_system=sl.COORDINATE_SYSTEM.IMAGE,
			sensors_required=True
			)

		for file in self._files2process:
			zed_camera = sl.Camera()
			input_file_path = str(Path(file).absolute())
			init_parameters.set_from_svo_file(input_file_path)
			err = zed_camera.open(init_parameters)
			if err != sl.ERROR_CODE.SUCCESS:
				sys.stdout.write(repr(err))
				zed_camera.close()
				print(f'Failed read svo file for view: {Path(input_file_path).stem}')
				exit()
			
			image_size = zed_camera.get_camera_information().camera_configuration.resolution
			self.width = image_size.width
			self.height = image_size.height
			self.fps = zed_camera.get_camera_information().camera_configuration.fps
			print(f'Processing {Path(input_file_path).stem}')
			print(f'Image size: {self.width}x{self.height}')
			print(f'FPS: {self.fps}')
			view_left_frame = sl.Mat(
				self.width,
				self.height, 
				sl.MAT_TYPE.U8_C3, 
				sl.MEM.CPU)
			view_right_frame = sl.Mat(
				self.width,
				self.height, 
				sl.MAT_TYPE.U8_C3, 
				sl.MEM.CPU)
			depth_frame = sl.Mat(
				self.width,
				self.height, 
				sl.MAT_TYPE.U8_C4, 
				sl.MEM.CPU)
			sensors_data = sl.SensorsData()
			runtime_parameters = sl.RuntimeParameters()
			runtime_parameters.confidence_threshold = 80
			Path(f"{output_folder}/{Path(input_file_path).stem}").mkdir(parents=True, exist_ok=True)
			writer4left = self.create_writer(f"{Path(input_file_path).stem}_left", f"{output_folder}/{Path(input_file_path).stem}")
			writer4right = self.create_writer(f"{Path(input_file_path).stem}_right", f"{output_folder}/{Path(input_file_path).stem}")
			today = datetime.now().strftime("%Y%m%d_%H%M%S")
			counter = 0
			svo_number_of_frames =  zed_camera.get_svo_number_of_frames()
			imu_data = open(f"{output_folder}/{Path(input_file_path).stem}/{Path(input_file_path).stem}_imu.csv", 'w')
			for counter in tqdm(range(svo_number_of_frames)):
				if zed_camera.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
					zed_camera.retrieve_image(view_left_frame, sl.VIEW.LEFT)
					zed_camera.retrieve_image(view_right_frame, sl.VIEW.RIGHT)
					zed_camera.retrieve_measure(depth_frame, sl.MEASURE.DEPTH)
					writer4left.write(view_left_frame.get_data()[:,:,0:3].astype(np.uint8))
					writer4right.write(view_right_frame.get_data()[:,:,0:3].astype(np.uint8))
					if zed_camera.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
						accelerometer = sensors_data.get_imu_data().get_linear_acceleration()
						gyroscope = sensors_data.get_imu_data().get_angular_velocity()
						imu_data.write(f"{counter}, {accelerometer[0]}, {accelerometer[1]}, {accelerometer[2]}, {gyroscope[0]}, {gyroscope[1]}, {gyroscope[2]}\n")
			writer4left.release()
			writer4right.release()
			zed_camera.close()

			
	def create_writer(self, _output_filename, _output_folder):
		"""
		Create a video writer object based on the specified output filename, output folder, and quality.

		Parameters:
		_output_filename (str): The name of the output video file.
		_output_folder (str): The path to the output folder where the video file will be saved.

		Returns:
		writer: The video writer object.

		"""
		video_file_processed = f"{_output_folder}/{_output_filename}.mp4"

		if self._MACHINE =='x86_64':
			writer = cv2.VideoWriter(
				video_file_processed,
				cv2.VideoWriter_fourcc(*'avc1'), int(self.fps), (self.width, self.height)
			)
		else:
			if self._codec == 'h265':
				writer = cv2.VideoWriter(f"appsrc ! video/x-raw, format=BGR ! queue ! \
				videoconvert ! video/x-raw,format=RGBA ! nvvidconv ! \
				nvv4l2h265enc bitrate=3000000 profile=0 control-rate=0 iframeinterval=100 ! \
				h265parse ! qtmux ! filesink location={video_file_processed} " , 0, int(self.fps), (self.width, self.height))
			if self._codec == 'h264':
				writer = cv2.VideoWriter(f"appsrc ! video/x-raw, format=BGR ! queue ! \
				videoconvert ! video/x-raw,format=RGBA ! nvvidconv ! \
				nvv4l2h264enc bitrate=3000000 profile=0 control-rate=0 iframeinterval=100 ! \
				h264parse ! qtmux ! filesink location={video_file_processed} " , 0, int(self.fps), (self.width, self.height))
		return writer
