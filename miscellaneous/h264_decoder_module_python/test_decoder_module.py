'''
    Copyright (C) <2018>  <Josue R. Cuevas>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import time
from h264module import *
from PIL import Image

decoder = init_h264dec("PATH_TO_FILE_TO_DECODE")
SAVE_TO = "PATH_WHERE_DECODED_FRAMES_CAN_BE_DUMPED"
DEFAULT_H = 720
DEFAULT_W = 1280
DEFAULT_C = 3

for i in range(20000):
    start = time.time()
    frame = get_frame(decoder, width=DEFAULT_W,
                      height=DEFAULT_H, channels=DEFAULT_C)
    end = time.time()

    # check if we can continue with the decoding
    if (frame.shape[0]) != DEFAULT_H:
        break

    print("Time taken to decode frame %d was: %4.5f ms" % (i, 1000 * (end - start)))
    print("Frame of dimensions: %d X %d x %d" %
          (frame.shape[0], frame.shape[1], frame.shape[2]))
    # check-image
    j = Image.fromarray(frame, mode='RGB')
    j.save(SAVE_TO+str(i)+".jpeg")

close_h264dec(decoder)
