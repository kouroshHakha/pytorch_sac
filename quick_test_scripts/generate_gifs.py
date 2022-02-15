from pprint import pprint
import imageio
import argparse
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str)
parser.add_argument('bname', type=str)
pargs = parser.parse_args()

folder = Path(pargs.folder)
output_file = folder / f'{pargs.bname}.gif'

filenames = []
for item in folder.iterdir():
    if item.is_file() and item.stem.startswith(pargs.bname):
        filenames.append(item)

# sort file names according to their string identifier
# look at the stem of the path, remove none-numbers, sort by intified version of number
filenames.sort(key=lambda f: int(re.sub('\D', '', f.stem)))

images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(output_file, images)