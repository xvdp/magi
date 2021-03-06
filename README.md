# Magi
*Magister Lt. teacher*

**STATUS: version 0.0.9 - passes tests,  WIP - currently in translation from private repos**

## Installation
Requires python > 3.4, testing on pytorch 1.7 - 1.9
```bash
git clone --recursive https://github.com/xvdp/magi # recursive as this requires some minor functions from xvdp/koreto
cd magi
pip install .
```

In progress, not guaranteed to be back compatible.
I'm translating an augmentation from untested private repos., and to make sure that I can stop rewriting the same code, generalizing at the same time. Should have comparable transforms to most augmentation pipleines shortly.


A collection of augmentation, training and dataloader wrappers for `pytorch`.

*Design intent, or why do we need another augmentation framework?*  I do.

1. world data is multimodal. Data loading and augmentation requires ad hoc code for each new data type or set of data types.

One may want to pass images alongside with annotations, sounds or graphs into a learner, or a collection of data may require to share parameters across multiple models. To handle generatilty, a container class inheriting from list, `Item`, with elements tagged with name, format, dtype, permits both flexibility and specificity. Datasets and Dataloaders output to Item, augmentation can interpret the datatypes and transform them accordingly.

2. Control of randomization of augmentation parameters. Leveraging torch.distributions transform parameters can both be randomized and included inside differentiable pipelines.


Parts:<br>
1. Features - data container `Item`
2. Agumentation - [Augumentations](AUGMENT.md)
3. Datasets and Dataloaders


## 1.Features: `Item()`

Dataset features are an open ended problem, a dataset may provide data with class names, regression targets, or a collection of data of different modes. <br>

To handle mulitmodal data, a feature class `Item` inheriting from `list`, with parallel lists tagging each item 'name', 'kind', 'dtype', 'form'.
Once augmented, an Item can be cast to list, stripping uneccessary information.

Datasets in this project output `__getitem__()` as Item()

For instance, getting an item from WIDER dataset
```python
>>> from magi.datasets import WIDER
>>> W = WIDER()
>>> data = W.__getitem__()
>>> data.keys
['names', 'kind', 'dtype', 'form']
>>> data.names
['image', 'bbox', 'name', 'blur', 'expression', 'illumination', 'invalid', 'occlusion', 'pose', 'index', 'wider_activity', 'wider_id', 'wordnet_id']
>>> data.kind
['data_2d', 'pos_2d', 'path', 'attr_id', 'attr_id', 'attr_id', 'attr_id', 'attr_id', 'attr_id', 'image_id', 'class_name', 'class_id', 'class_id']
['float32', 'float32', 'str', 'uint8', 'bool', 'bool', 'bool', 'uint8', 'bool', 'int', 'str', 'uint8', 'int']
>>> data.form
['NHCW', 'xywh', None, None, None, None, None, None, None, None, None, None, None]
>>> data.get(kind='path')
['/home/z/data/Face/WIDER/WIDER_train/images/37--Soccer/37_Soccer_Soccer_37_192.jpg']
>>> data.get(kind='pos_2d')
[tensor([[[[439.,  63.], [ 51.,  69.]], [[584., 148.], [ 55.,  68.]], [[680., 124.], [ 63.,  58.]], [[888.,  74.], [ 38.,  45.]]]])]
>>> data.get_indices(kind='pos_2d')
[1]
>>> data.form[1]
'xywh' # -> positional annotation format
```

## 2. Augmentation Transforms

Augumentation transforms nomenclature is based on the design of torchvision, located in folder `magi/transforms/`, with the intent of generalizing uses.

Transforms are classes which `__call__()` functionals. Functionals have a main transform wrapper generalized for 3 different purposes:
* dataset load - designed to minimize footprint, operations are where possible, in place
* display - cloning data at every step, allowing to trace and visualize augmentation
* differentiation - for backpropagation, triggered automatically


Transformations have class parameter `__type__` that specify to the type of action on the data: `IO`, `Compose`, `Appearance`, `Resizing`, `Affine`. Even though IO 'transforms' cannot be strictly considered as transforms, for simplicity they are built with the same syntax. For instance opening images as properly formated tensors and displaying them, can be done with simple calls.
```python
    from magi.transforms import *
    x = Open()(<path0>) # x tensor shape 1CHW, torch default dtype.
    y = Open()(<path1>)
    Show()([x,y])
```
Most parameter values for transformations (other than IO) can be either constant or randomized leveraging class `Values(a,b,distribution)` built over `torch.distributions`, and `Probs(p)` which serves as a Bernoulli mask. Any of the data dimensions can be probabilistic, i.e. an image batch may be augmented with separate parameters for sample, channel or even pixel. 

Transformations used in an augmentation pipleine or scripting envrornemnt can be called from the transform class or, if for instance are used inside a process requiring backprop, from the functional. Functionals for supported data kinds are suffixed, tagged and handled by the main functional transform wrapper in `functional_base`. For example, on affine transforms bounding boxes or paths on an image need to be transformed along images, transform `Rotate()` will call functional `rotate_tensor()` and `rotate_positions()`. New types of data that require transformation need to be tagged and an appropriate functional handler built.

Higher level transforms--handling data loaded from datasets or streams--are typechecked, through the `Item()` container class..

For a description of different augmentations: [Augumentations](AUGMENT.md)



<!-- ```python
    item = Item([[[0,1],[2,3]],[[1,2],[3,4]],  [125,125]], kind=["data","data", "id"], dtype=["float32", "float16", "int"])
    print(item, isinstance(item, list) # -> [[[0, 1], [2, 3]], [[1, 2], [3, 4]], [125, 125]] True

    item.to_torch(device="cuda")
    print(item) # returns each item with the assigned dype
    # -> [tensor([[0., 1.],
    #             [2., 3.]], device='cuda:0'), tensor([[1., 2.],
    #             [3., 4.]], device='cuda:0', dtype=torch.float16), tensor([125, 125], device='cuda:0', dtype=torch.int32)]
    print(item.keys) #-> ['kind', 'dtype']
    print(item.kind) #-> ['data', 'id']
    item.get(kind="data_2d") # returns
    #-> [tensor([[0., 1.],
    #           [2., 3.]], device='cuda:0'), tensor([[1., 2.],
    #           [3., 4.]], device='cuda:0', dtype=torch.float16)]

``` -->


## 2. datasets abd dataloading
Dataloader can pass transforms as dictionaries, for serialization.
Not ported yet.
### training
tbd


WIP
### Installation
Magi depends on submodules
```
git clone https://github.com/xvdp/magi --recursive && cd magi
python setup.py install

```
<!--when koreto updated
 git submodule foreach 'git pull origin main || :' -->