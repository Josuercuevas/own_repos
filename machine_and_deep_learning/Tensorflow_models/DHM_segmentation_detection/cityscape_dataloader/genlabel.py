'''
Copyright (C) <2017>  <Josue R. Cuevas>

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

import json
import os
import sys
sys.path.append('cocoapi/PythonAPI')
from pycocotools.coco import COCO
from pycocotools.mask import toBbox, frPyObjects
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#colors = plt.cm.hsv(np.linspace(0, 1, 8)).tolist()
colors = []
for c in xrange(8):
    colors.append(np.random.random((1, 3)).tolist()[0])

instance_label = ['person','rider','car','bus','truck','on rails','motorcycle','bicycle']
semantic_label = ['road', 'sidewalk','building','wall', 'fence', 'pole','traffic light', 'trafic sign', 'vegetation',
                  'terrain', 'sky','person','rider','car','bus','truck','on rails','motorcycle','bicycle' ]
def getAnns(filename, imgname):
    #I = cv2.imread(imgname)
    #fig, ax = plt.subplots(1)
    #ax.imshow(I)
    #ax = plt.gca()
    #ax.set_autoscale_on(False)
    semlabel = []
    inslabel = []
    bbs = []
    semsegs = []
    inssegs = []
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            jsonText = f.read()
            jsonDict = json.loads(jsonText)
            for key in jsonDict:
                leafDict = jsonDict['objects']
                for i in xrange(len(leafDict)):
                    polygon = leafDict[i]['polygon']
                    seg = []
                    for point in polygon:
                        seg.append(point[0])
                        seg.append(point[1])

                    #print rle
                    if leafDict[i]['label'] in semantic_label:
                        ids = semantic_label.index(leafDict[i]['label'])
                        semlabel.append(ids)
                        semsegs.append(seg)
                    if leafDict[i]['label'] in instance_label:
                        ids = instance_label.index(leafDict[i]['label'])
                        #print instance_label[ids]
                        #rle = frPyObjects([seg], jsonDict['imgHeight'], jsonDict['imgWidth'])[0]
                        #bb = toBbox(rle)
                        #bbs.append(bb)
                        inslabel.append(ids)
                        inssegs.append(seg)
                        #rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3],
                        #                         linewidth=1, edgecolor=colors[ids],
                        #                         facecolor='none')
                        #ax.add_patch(rect)
            #plt.show()
    return inslabel, inssegs, semlabel, semsegs
                #if key in self.__dict__:
                #    self.__dict__[key] = jsonDict[key]

if __name__ == "__main__":
    imgPath = 'CITYSCAPE/gtFine_trainvaltest/gtFine/train/aachen/aachen_000005_000019_gtFine_color.png'
    jsonpath = 'CITYSCAPE/gtFine_trainvaltest/gtFine/train/aachen/aachen_000005_000019_gtFine_polygons.json'
    getAnns(jsonpath, imgPath)
