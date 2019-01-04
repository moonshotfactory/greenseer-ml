import numpy as np
import argparse
import imageio
import os

from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--image-dir', required=True)
parser.add_argument('--background-image', default='background.jpg')
parser.add_argument('--num-train', default=70)
parser.add_argument('--num-val', default=20)
parser.add_argument('--num-test', default=10)
parser.add_argument('--output-dir', default='generated')
parser.add_argument('--lst-prefix', default='chess')
args = parser.parse_args()


def preview_merged(image_dir, background_image, foreground_images):
    background_image_path = os.path.join(image_dir, background_image)
    background = Image.open(background_image_path)
    all_boxes = []
    all_ids = []
    class_names = ['bishop', 'rook']
    id_dict = {'bishop.png': 0, 'rook.png': 1}

    for coordinates, name in foreground_images.items():
        print(name)
        image_path = os.path.join(image_dir, name)
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


    MIN_X = 100
    MAX_X = 2800
    MIN_Y = 100
    MAX_Y = 2800

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
    input_dir = os.path.expanduser(".")
    FILES_DICT = {p: glob.glob(os.path.join(input_dir, p + '*')) for p in ALL_PIECES}


    def generate_types():
        _types = []
        for i, p in enumerate(ALL_PIECES):
            name = _get_orig_piece(p)
            max_occurrences = MAX_DICT[name]
            num_occurrences = np.random.randint(1, max_occurrences + 1)
            _types += [i for _ in range(num_occurrences)]
        return _types
        
    def generate_coordinates():
        x = np.random.randint(MIN_X, MAX_X)
        y = np.random.randint(MIN_Y, MAX_Y)
        return (x, y)
            

    with open(lst_filepath, 'w') as fw:
        for i in range(num_examples):
            types = generate_types()

            fg_images = {}
            for t in enumerate(types):
                piece = ALL_PIECES_DICT[i]
                name = _get_orig_piece(piece)
                coordinates = generate_coordinates()
                fg_images[coordinates] = {0: 'bishop.png', 1: 'rook.png'}[t]

            img, all_boxes, all_ids, _ = preview_merged(
                args.image_dir, args.background_image, fg_images)
            filename = f'img_{i}.jpg'
            filepath = os.path.join(_dir, filename)
            imageio.imwrite(filepath, img)
            line = write_line(filename, img.shape, all_boxes, all_ids, i)
            fw.write(line)
