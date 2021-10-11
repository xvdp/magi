# Magi
*Magister Lt. teacher*

A collection of augmentation, training and dataloader wrappers for `pytorch` that I use on various projects, built this before torchvision supported transforms on tensors. This repo is an extraction of common methods from larger private project.

### features
Dataset features are an openended problem, a dataset may provide images with class names, regression targets, or a collection of data of different modes.
To address this openendednes this adds `DataItem` class, a feature that is both a list and can contain tags to aid the handling of data.   


### augmentation
Augmentation closely follows torchvision.transform design, with classes with callable methods. The main differences are that most of the implementations use pytorch native code and batches are built to handle any tagged data, positional annotations, provided that handlers for that data have been registered. <br> 

Extending pytorch convension, data batches are fed as tagged lists using `class DataItem` from datasets/features.py. ListDict behaves as a list and a dictionary,  each list element requires a corresponding element, which may be a tag or other data,  for each of the keys in DataItem. <br>
 e.g.
```python
# if image batch is in form
data = DataItem([tensor, tensor_list, indices], tags=["tensor2d", "positions2d", "indices"])
# where len(tensor) == len(list_of_tensor_annotations) == len(tensor_indices) == N, size of batch
out = magi.transforms.Rotation()(data)
# out -> DataItem([rotated_tensor, rotated_tensor_list, indices], tags=["tensor2d", "positions2d", "indices"]))
```
Given the wide variety of data, this is currently loosely typed. <br>
Handlers for how to interpret the different data are registered under transforms/handlers.py 

<!-- Tensors and lists of tensor annotations are rotated in the order defined in config, y or x dominant. <br>
Tensor lists can be of form [N,2,2] in the case of a box annotation or [N,2,M] in the case of paths. -->



### datasets abd dataloading
Dataloader can pass transforms as dictionaries, for serialization.

### training
tbd





WIP
### Installation
Magi depends on submodules
```
git clone https://github.com/xvdp/magi --recursive && cd magi
python setup.py install

焦げ koge
```
