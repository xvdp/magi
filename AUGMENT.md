# Data Transforms

## IO Transforms
**`__type__ = 'IO'`**

Not strictly 'augmentations' they have similar rules.

### Open()
### Save()
### Show()

## Composition Transforms
**`__type__ = 'Compose'`**
### Compose
### TwoCrop
### MultiCrop
### Fork
### Laplacian

## Randomization of transform parameters
All transforms, other than 'IO' and 'Compose' can be randomized. Randomization is performed by two classes, **`Values()`** and **`Probs()`** both of which derive from **`torch.distrbutions`** managing distribution properties from general Transform() arguments:
* `<value>: (int, float, tuple, list, tensor)` parameterizes transforms, where `<value>` represents any transform parameter, e.g. `mean` and `std` for `Normalize()`.<br>
When called with ` distribution=None `, or without secondary value kwarg values are constant.

 `Values()`, a managed wrapper to a subset of `torch.distributions`  defaulting to `Uniform()`.<br> 

**value kwargs available to transforms**:
*  `a, b, **kwargs`  distribution parameters kwargs of the distribution in question, e.g. `{'loc':0,'scale':2}` or `{'a':0,'b':2}`, the latter syntax is type checked and broadcasted. On transforms with more than one randomizable value (e.g. `Normalize(mean, center`)) the kwarg secondary value is suffixed to the name of the argument, e.g. `mean`, `mean_a`, `mean_c`...
*   `center: bool = True  ` centered distributions (Normal, Laplace, Gumbel) can reintepret 'a' and 'b' to be values at 3 standard deviations
*   `distribution: str = 'Uniform'`
*   `expand_dims: int, tuple = None  `   independent dims, eg. if `dims=0`, returns `N` random values; `dims=(0,1)` returns `N*C` values; `dims=(0,1,2,3)` returns `N*C*H*W` values. Each value modulates augmentation transform.  
*   `seed: int = None  `   calls `torch.manual_seed(seed)` # TODO: warn on multiple calls per session
*   `dtype, device`   presets the dtype and device - if they differ from the data, the Distribution is adjusted on call.

`Probs()` is a thin wrapper over `Values()` with a Bernoulli moudulation of the probablity of an augmentation taking place or not. Like Values(), Probs() can be applied per batch, sample, channel or any folllowing dimensions. kwargs to Probs are: 
* `p: int, float tensor`   Bernoulli probability
* `p_dims` redirects to expand_dims 

## Appearance Transforms
**`__type__ = 'Appearance'`**

### **Normalize()**
With alias  `MeanCenter()` standard normalization transform `(x-mean)/std`. `'mean'` and `'std'` can be probabilistic. 

### **UnNormalize()**
Inverse of normalize.

### **NormToRange()**
Linearly transform values to range between minimum and maximum. Most common use case of `NormToRange(minimum=0, maximum=1)`. `'minimum'` and `'maximum'` can be probabilistic.

### **Saturate()**
Changes saturation values, `Saturate(a=0, p=1)` converts to grayscale. a<0 inverts saturation, a>1 oversaturates. `(a=-1, b=2, distribution='Uniform', p=0.5, p_dims=0, expand_dims=(0,1)) ` outputs random distribution samples between inverse and over saturation over samples and channels, with 50% probability over sample.
```python
# Example Saturate with Categorical distribution over values (-2, 0, 3, 10) with 90% probability.
d = I.__getitem__()
img = merge_items((d,d,d,d))
Sat = Saturate(a=-2, b=3, c=10, d=0, p=0.9, p_dim=0, expand_dims=0, distribution="Categorical", for_display=True)
Show()(Sat(img))
```
<div align="center">
  <img width="100%" src=".github/Saturate_Categorical_abcd.png">
</div>

### **Gamma()**
Apply gamma. One probabilistic argument, target gamma `'a'`, one constant, source gamma, with default 2.2
## Affine Transforms


## Compose Transforms