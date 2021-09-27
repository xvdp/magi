# Magi
*Magister Lt. teacher*

A collection of augmentation, training and dataloader wrappers for `pytorch` that I use on various projects, built this before torchvision supported transforms on tensors. This repo is an extraction of common methods from larger private project.


### augmentation
Augmentation closely follows torchvision.transform design, with classes with callable methods. The main differences are that most of the implementations use pytorch native code and handle both data and positional annotations. <br> e.g in pseudocode <br>
```python
# if image batch is in form
batch = [tensor, tensor_list, indices]
# where len(tensor) == len(list_of_tensor_annotations) == len(tensor_indices) == N, size of batch
out = magi.transforms.Rotation()(batch)
# out = [rotated_tensor, rotated_tensor_list, indices]
```
Tensors and lists of tensor annotations are rotated in the order defined in config, y or x dominant. <br>
Tensor lists can be of form [N,2,2] in the case of a box annotation or [N,2,M] in the case of paths.



### dataloading
Dataloader can pass transforms as dictionaries, for serialization.

### training
tbd


Caveat: not thoroughly tested or optimized in multi-GPU settings


WIP