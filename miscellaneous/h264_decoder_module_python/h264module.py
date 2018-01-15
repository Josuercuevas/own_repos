'''
    Author: Josue R. Cuevas
    Date: 06/07/2017

    This modules is in charge of linking ffmpeg library
    and dependencies for decoding h264 bitstream files.
    It is an easy and fast implementation to extract RGB frames
    from encoded video with h264-standard format.

    For reference just check inter-process communication

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
import subprocess as sp
import numpy
import os.path
import re
import sys
import time
DEVNULL = open(os.devnull, 'wb')


def is_string(obj):
    """ Returns true if s is string or string-like object,
    compatible with Python 2 and Python 3."""
    try:
        return isinstance(obj, basestring)
    except NameError:
        return isinstance(obj, str)


def cvsecs(time):
    # Will convert any time into seconds.
    # Here are the accepted formats:
    # >>> cvsecs(15.4) -> 15.4 # seconds
    # >>> cvsecs( (1,21.5) ) -> 81.5 # (min,sec)
    # >>> cvsecs( (1,1,2) ) -> 3662 # (hr, min, sec)
    # >>> cvsecs('01:01:33.5') -> 3693.5  #(hr,min,sec)
    # >>> cvsecs('01:01:33.045') -> 3693.045
    # >>> cvsecs('01:01:33,5') #coma works too

    if is_string(time):
        if (',' not in time) and ('.' not in time):
            time = time + '.0'
        expr = r"(\d+):(\d+):(\d+)[,|.](\d+)"
        finds = re.findall(expr, time)[0]
        nums = list(map(float, finds))
        return (3600 * int(finds[0])
                + 60 * int(finds[1])
                + int(finds[2])
                + nums[3] / (10 ** len(finds[3])))

    elif isinstance(time, tuple):
        if len(time) == 3:
            hr, mn, sec = time
        elif len(time) == 2:
            hr, mn, sec = 0, time[0], time[1]
        return 3600 * hr + 60 * mn + sec

    else:
        return time

def get_setting(varname):
    """ Returns the value of a configuration variable. """
    gl = globals()
    if varname not in gl.keys():
        raise ValueError("Unknown setting %s" % varname)
    # Here, possibly add some code to raise exceptions if some
    # parameter isn't set set properly, explaining on how to set it.
    return gl[varname]


'''
 ffmpeg -i golden_multitask.mp4 -ss 00:00:20 -to 00:00:25 first_%d.jpg -ss 00:00:30 -to 00:00:35 sec_%d.jpg
'''

# Name of the binary file to be called
FFMPEG_BIN = "ffmpeg"
def init_h264dec(input_file, logfile=None, shifts_ms=None):

    extension = os.path.splitext(input_file)[1]
    times_stamps = dict()

    if logfile is not None:
        '''
            This will extract all the timestamps from all the frames
            inside the file if is needed only if the extension is not
            h264 or 264
        '''
        if (extension != '.h264') and (extension != '.264'):
            cmd = [FFMPEG_BIN, "-i", input_file]
            popen_params = {"bufsize": 10 ** 2,
                            "stdout": sp.PIPE,
                            "stderr": sp.PIPE,
                            "stdin": DEVNULL}

            '''
                Pipe containing the decoded frames extracted
                from the h264 video sequence
            '''
            print("Opening pipe ...")
            pic_pipe = sp.Popen(cmd, **popen_params)
            pic_pipe.stdout.readline()
            pic_pipe.terminate()
            infos = pic_pipe.stderr.read().decode('utf8')
            del pic_pipe

            lines = infos.splitlines()
            if "No such file or directory" in lines[-1]:
                raise IOError(("Decoder error: the file %s could not be found !\n"
                               "Please check that you entered the correct "
                               "path.") % input_file)

            # get duration (in seconds)
            times_stamps['duration'] = None

            try:
                keyword = ('Duration: ')
                # for large GIFS the "full" duration is presented as the last element in the list.
                index = 0
                line = [l for l in lines if keyword in l][index]
                match = re.findall("([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])", line)[0]
                times_stamps['duration'] = cvsecs(match)
            except:
                raise IOError(("Decoder error: failed to read the duration of file %s.\n"
                               "Here are the file infos returned by ffmpeg:\n\n%s")%(input_file, infos))

            # get the output line that speaks about video
            lines_video = [l for l in lines if ' Video: ' in l and re.search('\d+x\d+', l)]

            times_stamps['video_found'] = (lines_video != [])

            if times_stamps['video_found']:
                line = lines_video[0]
                match = re.search("( [0-9]*.| )[0-9]* fps", line)
                fps = float(line[match.start():match.end()].split(' ')[1])
                times_stamps['fps'] = fps
                times_stamps['frame_dur'] = 1.0/float(fps)


                """
                    extract the times from the log file and then determine how many frames we have to decode
                    to later set at fast seeker pipe with ffmpeg, +1, -1 second
                """
                time_formats_starters = []
                time_formats_finishers = []
                starting_times = []
                finish_times = []
                FIRST_TIME = True
                last_finish_point = 0
                indexer = 0
                for l in range(len(shifts_ms)):
                    time_sec = shifts_ms[l]/1000.0#seconds
                    if FIRST_TIME:
                        if(time_sec > 1.0):
                            starting_times.append(time_sec - 1)
                            if (time_sec+1.0) <= times_stamps['duration']:
                                finish_times.append(time_sec + 1.0)
                            else:
                                finish_times.append(times_stamps['duration'])#EOF
                        else:
                            # overall starter
                            starting_times.append(0.0)
                            finish_times.append(time_sec + 1.0)

                        last_finish_point = finish_times[indexer]
                        FIRST_TIME = False
                    else:
                        if last_finish_point < (time_sec-1.0):
                            if (time_sec > 1.0):
                                starting_times.append(time_sec - 1)
                                if (time_sec + 1.0) <= times_stamps['duration']:
                                    finish_times.append(time_sec + 1.0)
                                else:
                                    finish_times.append(times_stamps['duration'])  # EOF
                            else:
                                # overall starter
                                starting_times.append(0.0)
                                finish_times.append(time_sec + 1.0)

                            last_finish_point = finish_times[indexer]
                        else:
                            # we convered this already
                            continue


                    # print(starting_times[indexer], time_sec, finish_times[indexer])

                    # starting time
                    s_m, s_s = divmod(starting_times[indexer], 60)
                    s_h, s_m = divmod(s_m, 60)
                    # ending times
                    e_m, e_s = divmod(finish_times[indexer], 60)
                    e_h, e_m = divmod(e_m, 60)

                    time_formats_starters.append("%02d:%02d:%02d" % (s_h, s_m, s_s))
                    time_formats_finishers.append("%02d:%02d:%02d" % (e_h, e_m, e_s))

                    # print(time_formats_starters[indexer], time_formats_finishers[indexer])

                    indexer += 1

                frame_counter = ((finish_times[indexer-1] - starting_times[0]) * times_stamps['fps']) + 1
                times_stamps['n_frames'] = int(frame_counter)

            # print(times_stamps)
            # print(time_formats_starters)
            # print(time_formats_finishers)

            '''
                Command to be used as if we were on terminal
                the order of the arguments are:
                    1. Binary file to be called to execute FFMPEG
                    2. Input flag for ffmpeg
                    3. File to be used as input
                    4. Output file where we want the frames written
                    5. File name, in our case a Pipe
                    6. Pixel flag to tell ffmpeg how do we want the frames
                        to be decoded
                    7. Pixel format RGB24 -> 8bits per pixel
                    8. Decoding flag
                    9. Decoder to be used, in this case RAW, no formatting
                    10. end of arguments
            '''
            command = [FFMPEG_BIN, '-i', input_file,
                       '-ss', time_formats_starters[0], '-to', time_formats_finishers[indexer - 1], '-f', 'image2pipe',
                       '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
                       ]
        else:
            print("Is just elementary stream we cannot set timestamps ...")
            '''
                Command to be used as if we were on terminal
                the order of the arguments are:
                    1. Binary file to be called to execute FFMPEG
                    2. Input flag for ffmpeg
                    3. File to be used as input
                    4. Output file where we want the frames written
                    5. File name, in our case a Pipe
                    6. Pixel flag to tell ffmpeg how do we want the frames
                        to be decoded
                    7. Pixel format RGB24 -> 8bits per pixel
                    8. Decoding flag
                    9. Decoder to be used, in this case RAW, no formatting
                    10. end of arguments
            '''
            command = [FFMPEG_BIN, '-i', input_file, '-f', 'image2pipe',
                       '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
                       ]



        print("We are going to execute: ")
        print(command)

        pic_pipe = sp.Popen(command, bufsize=10 ** 8, stdout=sp.PIPE)

        if (extension != '.h264') and (extension != '.264'):
            return pic_pipe, times_stamps
        else:
            return pic_pipe, None

        # sys.exit(0)
    else:
        '''
            Command to be used as if we were on terminal
            the order of the arguments are:
                1. Binary file to be called to execute FFMPEG
                2. Input flag for ffmpeg
                3. File to be used as input
                4. Output file where we want the frames written
                5. File name, in our case a Pipe
                6. Pixel flag to tell ffmpeg how do we want the frames
                    to be decoded
                7. Pixel format RGB24 -> 8bits per pixel
                8. Decoding flag
                9. Decoder to be used, in this case RAW, no formatting
                10. end of arguments
        '''
        command = [FFMPEG_BIN, '-i', input_file, '-f', 'image2pipe',
                   '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-']
        pic_pipe = sp.Popen(command, bufsize=10**5, stdout=sp.PIPE)

        return pic_pipe, None


def close_h264dec(pic_pipe):
    pic_pipe.terminate()
    del pic_pipe


def get_frame(pic_pipe, width=1280, height=720, channels=3,
              need_process=False):
    # read width*height*channels bytes (= 1 frame)
    rgb_frame = pic_pipe.stdout.read(width * height * channels)

    # transform to numpy array to be used in tensorflow
    frame = numpy.fromstring(rgb_frame, dtype='uint8')

    if not need_process:
        '''
            we don't need more data from the pipe, so we just
            flush it. This could be info needed by ffmpeg and not
            for us to use.
        '''
        pic_pipe.stdout.flush()
        return frame

    if frame.shape[0] < (width*height*channels):
        print("No more frames to be decoded from this stream...")
        return frame

    frame.shape = (height, width, channels)# = frame.reshape([height, width, channels])

    '''
        we don't need more data from the pipe, so we just
        flush it. This could be info needed by ffmpeg and not
        for us to use.
    '''
    pic_pipe.stdout.flush()

    return frame
