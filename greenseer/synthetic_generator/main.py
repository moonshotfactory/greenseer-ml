import numpy as np
import argparse
import imageio
import os

from PIL import Image

# Usage: python greenseer/synthetic_generator/main.py --input-dir ./data/resized-labeled-images --output-dir ./data/syn-gen-images
# Followed by 
# python greenseer/synthetic/im2rec.py data/syn-gen-images/chess_train data/syn-gen-images/images/train --no-shuffle --pass-through --pack-label

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", required=True)
parser.add_argument('--output-dir', default='generated')
parser.add_argument('--lst-prefix', default='chess')
parser.add_argument('--num-train', default=7)
parser.add_argument('--num-val', default=2)
parser.add_argument('--num-test', default=1)

args = parser.parse_args()


MIN_X = 100
MAX_X = 2500
MIN_Y = 100
MAX_Y = 2500


PIECES = ['rook', 'knight', 'bishop', 'king', 'queen', 'pawn']
MAX_IMAGES_PER_CATEGORY = [2, 2, 2, 1, 1, 8]
MAX_DICT = {piece: MAX_IMAGES_PER_CATEGORY[i] for i, piece in enumerate(PIECES)}


PIECES_BLACK =[piece + '-black' for piece in PIECES] 
PIECES_WHITE =[piece + '-white' for piece in PIECES] 
ALL_PIECES = PIECES_BLACK + PIECES_WHITE
ALL_PIECES_DICT = {i: piece for i, piece in enumerate(ALL_PIECES)}

def _get_orig_piece(name):
    return name.split('-')[0]

import glob
import json

input_dir = args.input_dir 
FILES_DICT = {p: glob.glob(os.path.join(input_dir, p + '*')) for p in ALL_PIECES}
BG_DICT = {'background': glob.glob(os.path.join(input_dir, 'background*'))}

print(json.dumps(FILES_DICT, indent=4))
print(json.dumps(BG_DICT, indent=4))

# import sys
# sys.exit(1)



def preview_merged(background_images, foreground_images):
    all_boxes = []
    all_ids = []

    class_names = ['rook', 'knight', 'bishop', 'king', 'queen', 'pawn']
    id_dict = {name: i for i, name in enumerate(class_names)}

    background_image_path = np.random.choice(background_images)
    background = Image.open(background_image_path)

    for coordinates, image_path in foreground_images.items():
        image_filename = os.path.basename(image_path)
        name = image_filename.split('-')[0]

        foreground = Image.open(image_path)

        x, y = coordinates
        w, h = foreground.size
        label = [x, y, x + w, y + h]
        _id = id_dict[name]
        all_boxes.append(np.array(label))
        all_ids.append(_id)
        background.paste(foreground, coordinates, foreground)
    return np.array(background), np.array(all_boxes), np.array(all_ids), np.array(class_names)


def write_line(img_path, im_shape, boxes, ids, idx):
    h, w, c = im_shape
    # for header, we use minimal length 2, plus width and height
    # with A: 4, B: 5, C: width, D: height
    A = 4
    B = 5
    C = w
    D = h
    # concat id and bboxes
    labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
    # normalized bboxes (recommanded)
    labels[:, (1, 3)] /= float(w)
    labels[:, (2, 4)] /= float(h)
    # flatten
    labels = labels.flatten().tolist()
    str_idx = [str(idx)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_labels = [str(x) for x in labels]
    str_path = [img_path]
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    return line




def generate_types():
    _types = []
    for i, p in enumerate(ALL_PIECES):
        name = _get_orig_piece(p)
        max_occurrences = MAX_DICT[name]
        num_occurrences = np.random.randint(0, max_occurrences + 1)
        _types += [i for _ in range(num_occurrences)]
    return _types
    
def generate_coordinates():
    x = np.random.randint(MIN_X, MAX_X)
    y = np.random.randint(MIN_Y, MAX_Y)
    return (x, y)


num_examples_dict = {
    'train' : int(args.num_train),
    'val' : int(args.num_val),
    'test' : int(args.num_test),
}

for part in ['train', 'val', 'test']:
    _dir = os.path.join(args.output_dir, 'images', part)
    if not os.path.isdir(_dir):
        os.makedirs(_dir)

    lst_filename = f'{args.lst_prefix}_{part}.lst'
    lst_filepath = os.path.join(args.output_dir, lst_filename)
    num_examples = num_examples_dict[part]
        

    with open(lst_filepath, 'w') as fw:
        for i in range(num_examples):
            types = generate_types()

            fg_images = {}
            # import ipdb; ipdb.set_trace()
            for t in types:
                piece = ALL_PIECES_DICT[t]
                coordinates = generate_coordinates()
                # import ipdb; ipdb.set_trace()
                image_filename = np.random.choice(FILES_DICT[piece])
                fg_images[coordinates] = image_filename
                # import ipdb; ipdb.set_trace()

            img, all_boxes, all_ids, _ = preview_merged(list(BG_DICT.values())[0], fg_images)
            filename = f'img_{i}.jpg'
            filepath = os.path.join(_dir, filename)
            imageio.imwrite(filepath, img)
            line = write_line(filename, img.shape, all_boxes, all_ids, i)
            fw.write(line)
