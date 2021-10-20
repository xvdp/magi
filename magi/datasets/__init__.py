"""@xvdp

datasets
dataloaders

The purpose of a dataloader is to feed data to learners. 
Dataloaders are open ended problem. To make them as versatile as possible, magi datasets
return data as magi.Item type, which can be used or cast as a list
or can contain keys per list item.


Data may come in many different forms, from the simplest vision classification datasets
    [(NCHW)ImageTensors2d, (N)Labels], eg. ImageNet
to data with bounding box annotations,
    [(NCHW)ImageTensors2d, (NM42)List(N)ofList(M)ofBoxesBox2d, (N)Labels], eg. Wider or COCO
to data with segmentation masks
    [(NCHW)ImageTensors2d, (NML2)List(N)ofList(M)ofPaths(L), (N)Labels], eg. Wider or COCO
to data with lower or higher dimensions, (NCL)Tensors1d, or (NCHWD)Tensors3d,
to unlabelled data,(N)Labels, or data with multiple labels, (N)Labels, (N)Labels1
to data with labels per tesnsor element (NCHW)Lablels
to data with Sparse Tensor Labeling, or Adjacency Graphs.
from curated finite labeled datasets to open graphs.
"""
from .datasets import *
from .wider import WIDER
from .imagenet import ImageNet
