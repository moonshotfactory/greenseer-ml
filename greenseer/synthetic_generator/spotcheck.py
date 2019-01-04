import argparse
from gluoncv import utils as gcv_utils
import numpy as np
from gluoncv.data import RecordFileDetection
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('record_file')
args = parser.parse_args()

CLASS_NAMES = ['rook', 'knight', 'bishop', 'king', 'queen', 'pawn']


# USAGE: python greenseer/synthetic_generator/spotcheck.py ./data/syn-gen-images/chess_train.rec

# Load record file from ".rec" and ".idx" files
record_dataset = RecordFileDetection(args.record_file, coord_normalized=True)
num_records = len(record_dataset)
print(num_records)

# Focus on a random record
index = np.random.randint(0, num_records)
random_record = record_dataset[index]
img, label = random_record
print('Type of image', type(img))
print('Type of label', type(label))

# Display bounding boxes together with labels
ax = gcv_utils.viz.plot_bbox(img, label[:, :-1], labels=label[:, -1], class_names=CLASS_NAMES)
plt.show(ax)

