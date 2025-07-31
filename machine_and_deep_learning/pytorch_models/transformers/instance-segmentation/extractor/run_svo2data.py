import argparse
from svo2data import SVO2Data


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--source', help='Input source. A .svo2 file or folder.')
	args = parser.parse_args()
	svo2data = SVO2Data(args)
	svo2data.process()