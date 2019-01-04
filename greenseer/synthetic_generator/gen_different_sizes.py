import os
import glob
import argparse


from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('source_directory')
parser.add_argument('destination_directory')
args = parser.parse_args()

def resize_images(source_directory, destination_directory, pattern="*.png"):
    if not os.path.isdir(destination_directory):
        os.makedirs(destination_directory)
    resize_factors = [1.0, 1.5, 1.75, 2.0, 2.25, 2.5]
    files = glob.glob(os.path.join(source_directory, pattern)) 
    for filename in files:
        basename = os.path.basename(filename)
        image = Image.open(filename)
        for resize_factor in resize_factors:
            new_size = [int(float(e) / resize_factor) for e in image.size]
            resized_image = image.resize(new_size)
            parts = basename.split(".")

            new_basename = ".".join(parts[:-1] + ["{}x".format(resize_factor)] + parts[-1:])
            new_filename = os.path.join(destination_directory, new_basename)

            resized_image.save(new_filename)

    print(len(files))




def main(args):
    resize_images(args.source_directory, args.destination_directory)


# Usage: python greenseer/synthetic_generator/gen_different_sizes.py ./data/new-labeled-images ./data/resized-labeled-images
if __name__ == "__main__":
    main(args)
