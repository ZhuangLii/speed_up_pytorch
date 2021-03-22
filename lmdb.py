import cv2
import lmdb
import hashlib
import numpy as np


# --------read image ---------- #
db = lmdb.open(db_path, readonly=True, map_size=int(50e9))
txn = db.begin(write=False)

def imread(txn, path):
    key = hashlib.md5(path.encode()).digest()
    img_buffer = txn.get(key)
    img_buffer = np.frombuffer(img_buffer, np.uint8)
    img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
    return img

# --------create lmdb --------- #

import lmdb
import cv2
import numpy as np
import os 
import hashlib
import functools

from glob import glob
from fire import Fire
from tqdm import tqdm
from multiprocessing import Pool

def worker(video_name):
    image_names = glob(video_name+'/*')
    kv = {}
    for image_name in image_names:
        img = cv2.imread(image_name)
        _, img_encode = cv2.imencode('.jpg', img)
        img_encode = img_encode.tobytes()
        kv[hashlib.md5(image_name.encode()).digest()] = img_encode
    return kv

def create_lmdb(data_dir, output_dir, num_threads):
    video_names = glob(data_dir+'/*')
    video_names = [x for x in video_names if os.path.isdir(x)]
    db = lmdb.open(output_dir, map_size=int(50e9))
    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(
            functools.partial(worker), video_names), total=len(video_names)):
            with db.begin(write=True) as txn:
                for k, v in ret.items():
                    txn.put(k, v)

if __name__ == '__main__':
    Fire(create_lmdb)
