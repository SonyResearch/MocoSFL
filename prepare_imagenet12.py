import os
from glob import glob
import shutil
import cv2
import math

if __name__ == '__main__':
    os.system("git clone git@github.com:zlijingtao/imagnet-12.git")
    os.system("mv imagnet-12 ./data/")
