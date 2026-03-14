from os import path
from pathlib import Path
from tqdm import tqdm

DATA_PATH = Path('data1')


def digitize(path: Path):
    pass


def main():

    # find all png in data path
    png_files = list(path.glob('*.png'))

    for png_path in tqdm(png_files):
        digitize(png_path)


if __name__ == "__main__":
    main()
