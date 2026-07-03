"""Download the cats_and_dogs_filtered dataset used by HW2 (CNN part).

The dataset (~68 MB) is fetched at runtime and unzipped next to this file; it is
git-ignored and never committed.  Structure after download::

    cats_and_dogs_filtered/
        train/{cats,dogs}/*.jpg      (1000 each)
        validation/{cats,dogs}/*.jpg (500 each)
"""
import os
import urllib.request
import zipfile

URL = "https://storage.googleapis.com/tensorflow-1-public/course2/cats_and_dogs_filtered.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(HERE, "cats_and_dogs_filtered.zip")
DATA_DIR = os.path.join(HERE, "cats_and_dogs_filtered")


def download():
    if os.path.isdir(DATA_DIR):
        print(f"Dataset already present at {DATA_DIR}")
        return DATA_DIR
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    print("Unzipping ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(HERE)
    os.remove(ZIP_PATH)
    print(f"Done -> {DATA_DIR}")
    return DATA_DIR


if __name__ == "__main__":
    download()
