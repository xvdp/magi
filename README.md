# Magi
*Magister Lt. teacher*

A collection of augmentation, training and dataloader wrappers for `pytorch` that I use on various projects, built this before torchvision supported transforms on tensors. This repo is an extraction of common methods from larger private project.

### features
Dataset features are an openended problem, a dataset may provide data with class names, regression targets, or a collection of data of different modes. <br>
To address this openendednes this adds an addressable list, `class Item(list)` a feature that is both a list and can contain tags to aid the handling of data.<br>
Item can be cast to `list(Item)` or be used to carry any kind of structured data.
For example:

```python
item = Item([[[0,1],[2,3]],[[1,2],[3,4]],  [125,125]], meta=["data","data", "id"], dtype=["float32", "float16", "int"])
print(item, isinstance(item, list) # -> [[[0, 1], [2, 3]], [[1, 2], [3, 4]], [125, 125]] True

item.to_torch(device="cuda")
print(item) # returns each item with the assigned dype
# -> [tensor([[0., 1.],
#             [2., 3.]], device='cuda:0'), tensor([[1., 2.],
#             [3., 4.]], device='cuda:0', dtype=torch.float16), tensor([125, 125], device='cuda:0', dtype=torch.int32)]
print(item.keys) #-> ['meta', 'dtype']
print(item.meta) #-> ['data', 'id']
item.get("meta", "data") # returns
#-> [tensor([[0., 1.],
#           [2., 3.]], device='cuda:0'), tensor([[1., 2.],
#           [3., 4.]], device='cuda:0', dtype=torch.float16)]

```



### augmentation
Augmentation closely follows torchvision.transform design, with classes with callable methods. The main differences are that most of the implementations use pytorch native code and batches are built to handle any tagged data, positional annotations, provided that handlers for that data have been registered. <br> 

Extending pytorch convension, data batches are fed as tagged lists using `class Item` from datasets/features.py. ListDict behaves as a list and a dictionary,  each list element requires a corresponding element, which may be a tag or other data,  for each of the keys in Item. <br>
 e.g.
```python
# if image batch is in form
data = Item([tensor, tensor_list, indices], tags=["tensor2d", "positions2d", "indices"])
# where len(tensor) == len(list_of_tensor_annotations) == len(tensor_indices) == N, size of batch
out = magi.transforms.Rotation()(data)
# out -> Item([rotated_tensor, rotated_tensor_list, indices], tags=["tensor2d", "positions2d", "indices"]))
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
