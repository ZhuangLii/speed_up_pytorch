import os
import torch
import types
import collections
import numpy as np
# import cupy as cp
from os.path import join
from random import shuffle
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.ops as ops



class D2InputIterator(object):
    def __init__(self, ann_root, img_root, target_transform=None, batch_size=8, device_id=0, num_gpus=1, is_val=False):
        self.ann_root = ann_root
        self.img_root = img_root
        self.batch_size = batch_size
        self.target_transform = target_transform
        subfolders = os.listdir(ann_root)
        self.anns = []
        for folder in subfolders:
            if not is_val and not folder.startswith("0008"):
                txt_files = os.listdir(join(ann_root, folder))
                self.anns.extend([join(folder, txt) for txt in txt_files])
            elif is_val and folder.startswith("0008"):
                txt_files = os.listdir(join(ann_root, folder))
                self.anns.extend([join(folder, txt) for txt in txt_files])
        self.n = len(self.anns)
        # 注意: 每个迭代器的读取的 data以及gt 根据使用的gpu 数量而决定，每个gpu读取数据的一部分
        self.anns = self.anns[self.n * device_id // num_gpus: self.n * (device_id+1) // num_gpus]

        labels = ["__background__",
                  "car", "van", "bus", "truck", "person", "bicycle", "motorcycle", "open-tricycle",
                  "closed-tricycle", "forklift", "large-block", "small-block"]
        self._classes_names = labels
        self.ann_map = {item: i for i, item in enumerate(labels)}

    def __iter__(self):
        self.i = 0
        shuffle(self.anns)
        return self

    def __len__(self):
        return self.n

    def __next__(self):
        batch = []
        labels = []
        boxes = []

        if self.i >= self.n:
            raise StopIteration

        for _ in range(self.batch_size):
            ann_file = self.anns[self.i]
            img_file = ann_file.replace(".xml", ".mp4/img1")
            img_file = img_file.replace(".txt", ".jpg")
            with open(join(self.ann_root, ann_file), "r") as f:
                lines = f.readlines()
            img = open(join(self.img_root, img_file), 'rb')
            bboxs, ids = [], []
            for line in lines:
                frame_num, _, label, x, y, w, h = line.rstrip().split(",")
                x, y, w, h = list(map(float, [x, y, w, h]))
                bboxs.append([x, y, x + w, y + h])
                ids.append([self.ann_map[label]])
            batch.append(np.frombuffer(img.read(), dtype=np.uint8))
            ids = torch.tensor(ids, dtype=torch.uint8)
            bboxs = torch.tensor(bboxs, dtype=torch.float32)
            bboxs, ids = self.target_transform(bboxs, ids)
            labels.append(ids.numpy())
            boxes.append(bboxs.numpy())
            self.i = (self.i + 1) % self.n
        # 注意 batch_size (n,3,h,w), labels_size (n, total_anchors * 1) 1 for class, boxes_size (n, total_anchors, 4) 4 for x,y,w,h, 统一输出格式，不能因为一张图片中检测目标数量的不同，而labels和boxes 的 shope 不同
        return (batch, labels, boxes)
    next = __next__


class D2Pipeline(Pipeline):
    def __init__(self, resize, batch_size, num_threads, device_id, external_data):
        super(D2Pipeline, self).__init__(batch_size,
                                                     num_threads,
                                                     device_id,
                                                     seed=12,
                                                     exec_async=False,
                                                     exec_pipelined=False)
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.input_boxes = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        # 自定义的函数只能在cpu上运行
        # self.grid = ops.PythonFunction(function=grid2x2, num_outputs=4)
        self.resize = ops.Resize(device="gpu",
                                 resize_x=resize,
                                 resize_y=resize,
                                 interp_type=types.INTERP_LINEAR)
        self.external_data = external_data
        self.iterator = iter(self.external_data)

    def define_graph(self):
        # 定义graph 包含3个输入，分别叫做 self.jpegs, self.labels 和 self.boxes
        # self.images 经过decode 和 resize 输出
        # self.labels 和 self.boxes直接输出
        # transform 也是在这个里面做的，参考dali 的 transform
        self.jpegs = self.input()
        self.labels = self.input_label()
        self.boxes = self.input_boxes()
        images = self.decode(self.jpegs)
        images = self.resize(images.gpu())
        return (images, self.labels, self.boxes)

    def iter_setup(self):
        try:
            images, labels, boxes = self.iterator.next()
            self.feed_input(self.jpegs, images)
            self.feed_input(self.labels, labels)
            self.feed_input(self.boxes, boxes)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration


def create_dataloder(ann_root,
                     img_root,
                     resize,
                     batch_size,
                     device_id=0,
                     num_gpus=1,
                     num_threads=2,
                     target_transform=None,
                     is_val=False):
    #1.首先自定义一个迭代器来读取自己的数据
    eii = D2InputIterator(ann_root=ann_root,
                          img_root=img_root,
                          batch_size=batch_size,
                          device_id=device_id,
                          num_gpus=num_gpus,
                          target_transform=target_transform,
                          is_val=is_val)
    #2.定义一个 管道 继承于dali 的 pipeline 类
    pipe = D2Pipeline(resize=resize,
                            batch_size=batch_size,
                            num_threads=num_threads,
                            device_id=device_id,
                            external_data=eii)

    pii = DALIGenericIterator(pipe,
                            output_map=["img","labels","boxes"],
                            size=len(eii),
                            dynamic_shape=True,
                            last_batch_padded=True,
                            fill_last_batch=False)
    return pii


if __name__ == '__main__':
    batch_size = 32
    num_gpus = 1
    num_threads = 8
    epochs = 1
    img_root = "/VisualGroup/share/data/didi/mot_format_didi_data/"
    ann_root = "/root/D2DetectAnn/"
    # create_dataloader 返回 DALIGenericIterator 迭代器，用法和 pytorch 的dataloader一样，见 174 行
    pii = create_dataloder(ann_root,
                           img_root,
                           resize=360,
                           batch_size=4,
                           device_id=0,
                           num_gpus=1,
                           num_threads=1,
                           is_val=True)

    for e in range(epochs):
        for i, data in enumerate(pii):
            imgs = data[0]["img"]
            labels = data[0]["labels"]
            boxes = data[0]["boxes"]
            print("epoch: {}, iter {}".format(e, i), imgs.shape, labels.shape)
        pii.reset()
   #######
   #主要参考：https://blog.csdn.net/weixin_42028608/article/details/105564060
   #######
