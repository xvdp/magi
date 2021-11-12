
"""@xvdp
"""
from typing import Union
import logging
import numpy as np
import torch
from koreto import Col
from .torch_util import get_broadcastable

_torchable = (int, float, list, tuple, np.ndarray, torch.Tensor)


# pylint: disable=no-member
def to_saturation(x: torch.Tensor, saturation: Union[_torchable], axis: int = 1) -> torch.Tensor:
    """ lerps grayscale(x) with x by alpha
    Args
        x           tensor to saturate
        saturation  (int,float,list, tensor), could have per channel or sample values
        axis        int [1] channels axis
    """
    if not isinstance(saturation, (int, float)):
        saturation = get_broadcastable(saturation, other=x, axis=axis)

    if not torch.any(saturation):
        return to_grayscale(x, axis=axis)
    return torch.lerp(to_grayscale(x, axis=axis), x, saturation)

def to_grayscale(x: torch.Tensor, axis :int = 1) -> torch.Tensor:
    """ RGB to Grayscale
    x       torch.tensor
    axis    (int [1]) channels axis
    """
    shape = x.shape
    if shape[axis] == 3:
        x = x.mul(get_broadcastable([0.2989, 0.5870, 0.1140], other=x,
                                    axis=axis)).sum(axis=axis, keepdims=True)
    else:
        logging.warning(f"{Col.YB}expected RGB tensor over axis {axis}, found shape {tuple(x.shape)}, averaging...{Col.AU}")
        x = x.mean(axis=axis, keepdims=True)

    return x.broadcast_to(shape)



# from .random_util import normal, uniform

# # pylint: disable=no-member
# def get_random_color(tensor: torch.Tensor, distribution: str, independent: int=1, grad: bool=None) -> torch.Tensor:
#     """returns distribution of Lab values
#     Args:
#         tensor  (torch tensor)
#         distribution (str)  "imagenet": imagenet distribution (approx laplacian in Lab space)
#                             "normal":   normal distribution in range of imagenet values
#                             "uniform":  uniform distribution in range of imagenet
#         independent (int, default 1), 0, single sample for batch, 1, sample per image
#     """
#     assert distribution in ("imagenet", "normal", "uniform")
#     size = 1
#     dtype = tensor.dtype
#     device = tensor.device
#     grad = tensor.requires_grad if grad is None else grad

#     lab = Lab(dtype, device, grad)
#     size = len(tensor)
#     if not independent:
#         size = 1

#     return lab.like(size=size, distribution=distribution)

# class Lab:
#     """ Full (14M) ImageNet Color distributions precomuted
#     hname = '/media/z/Elements/data/ImageNet/stats.h5'
#     with h5py.File(hname) as stats:
#         lab_means = stats['lab_means'][:]
#     """
#     def __init__(self, dtype=None, device=None, grad=None):

#         self.dtype = config.DTYPE if dtype is None else dtype
#         self.device = config.DEVICE if device is None else device
#         self.grad = config.GRAD if grad is None else grad

#         self.Lab_mean = torch.tensor([51.05476929, 0.4165136, 8.72816564],
#                                      dtype=self.dtype, device=self.device, requires_grad=False)
#         # lab_means.max(0) - lab_means.min(0)
#         self.Lab_range = torch.tensor([100.0, 176.04899673, 194.8944961],
#                                       dtype=self.dtype, device=self.device, requires_grad=False)
#         self.Lab_max = torch.tensor([100., 95.57650759, 90.09804096],
#                                     dtype=self.dtype, device=self.device, requires_grad=False)
#         # lab_means.min(0)
#         self.Lab_min = torch.tensor([0., -80.47248913, -104.79645514],
#                                     dtype=self.dtype, device=self.device, requires_grad=False)
#         self. __len__ = 14177592

#     def like(self, size=1, distribution="imagenet"):
#         """Generate random distribution in Lab space
#         Args:
#             size    (int) number of samples
#             distribution    (str)
#                                 imagenet: matching ImageNet CIELab distribution
#                                 normal: normal distribution in the range of Imagenet
#                                 uniform: uniform distribution in the range of Imagenet
#         """
#         if distribution == "imagenet":
#             _binpick = torch.randint(low=0, high=14177592, size=(size, 3, 1),
#                                      dtype=self.dtype, device=self.device, requires_grad=False)

#             # print(f"binpick {_binpick}, {_binpick.shape}")
#             # print(f"lab_cumsum {self.lab_cumsum.shape}")
#             # print(f"rand {torch.abs(self.lab_cumsum - _binpick).shape}")
#             lab_rand = torch.argmin(torch.abs(_binpick - self.lab_cumsum),
#                                     dim=2).to(dtype=self.dtype, device=self.device)

#             # print(f"lab_rand {lab_rand}")
#             dist = lab_rand* self.Lab_range/256.0 + self.Lab_min
#             # print(f"dist {dist}")

#         elif distribution == "normal":
#             _l = torch.clamp(normal(50., 13., size, dtype=self.dtype,
#                                     device=self.device, grad=False), 0.0, 100.)
#             _a = torch.clamp(normal(0.4, 12., size, dtype=self.dtype,
#                                     device=self.device, grad=False), -80.0, 95.)
#             _b = torch.clamp(normal(8.7, 13., size, dtype=self.dtype,
#                                     device=self.device, grad=False), -104.0, 90.)
#             dist = torch.stack((_l, _a, _b)).t()

#         else: #"uniform"
#             _l = uniform(0.0, 100., size, dtype=self.dtype, device=self.device, grad=False)
#             _a = uniform(-80.0, 95, size, dtype=self.dtype, device=self.device, grad=False)
#             _b = uniform(-104., 90., size, dtype=self.dtype, device=self.device, grad=False)
#             dist = torch.stack((_l, _a, _b)).t()
#         return dist

#     @property
#     def lab_cumsum(self):
#         """
#         Precomputed cum sum of 8bit histogram for ImageNet L,a,b
#         Lab_dist = np.histogram(lab_means, bins=256)
#         np.cumsum(Lab_dist[0])
#         """
#         labsum = torch.tensor([[[27, 137, 429, 964, 1746, 830,
#                                  4216, 5775, 7490, 9368, 11480, 13611,
#                                  15940, 18358, 20968, 23654, 26384, 29346,
#                                  32280, 35344, 38525, 41893, 45245, 49109,
#                                  53118, 57319, 62004, 66845, 71876, 77389,
#                                  83119, 89069, 95500, 102388, 109420, 116711,
#                                  124614, 132890, 141499, 150896, 160494, 170603,
#                                  181037, 192342, 204144, 216437, 229360, 242969,
#                                  257249, 272213, 287990, 304761, 321784, 339492,
#                                  358343, 377835, 398370, 419901, 442741, 466394,
#                                  490768, 516490, 543252, 571285, 600644, 631296,
#                                  663494, 696859, 731707, 767819, 805484, 845086,
#                                  886386, 929390, 973976, 1020635, 1069264, 1119767,
#                                  1172557, 1227032, 1283698, 1342543, 1403420, 1466907,
#                                  1533322, 1601539, 1672226, 1745527, 1821726, 1900245,
#                                  1981634, 2065879, 2152818, 2242854, 2336077, 2432198,
#                                  2531085, 2633031, 2738157, 2846645, 2958325, 3073479,
#                                  3191427, 3312615, 3436915, 3564888, 3696075, 3830690,
#                                  3968046, 4108146, 4251445, 4398408, 4548276, 4700620,
#                                  4856795, 5015501, 5175990, 5338870, 5504122, 5672003,
#                                  5841509, 6011970, 6184495, 6358914, 6534437, 6709155,
#                                  6884528, 7061402, 7238539, 7415645, 7592568, 7768020,
#                                  7941952, 8114407, 8285080, 8454340, 8620273, 8783889,
#                                  8945173, 9102583, 9256679, 9406935, 9552953, 9695364,
#                                  9833785, 9968176, 10098308, 10223524, 10345022, 10462311,
#                                  10574954, 10683595, 10788220, 10888922, 10986555, 11080189,
#                                  11170625, 11257092, 11340340, 11420815, 11499051, 11573857,
#                                  11646429, 11715949, 11782898, 11847778, 11910426, 11971057,
#                                  12030022, 12087166, 12142420, 12195710, 12247092, 12297293,
#                                  12345993, 12394092, 12440100, 12485251, 12528906, 12571194,
#                                  12612871, 12652943, 12692436, 12730781, 12767740, 12804095,
#                                  12839883, 12875115, 12909395, 12943209, 12976078, 13008560,
#                                  13040129, 13071452, 13101996, 13132342, 13162301, 13191266,
#                                  13219808, 13248100, 13276289, 13303577, 13330733, 13357514,
#                                  13383740, 13409558, 13435134, 13460347, 13485013, 13509659,
#                                  13533863, 13557704, 13581178, 13604681, 13627793, 13650870,
#                                  13673540, 13695944, 13718091, 13739903, 13761317, 13782344,
#                                  13803174, 13823888, 13844113, 13863833, 13883353, 13902265,
#                                  13921094, 13939900, 13957693, 13974967, 13991577, 14007672,
#                                  14023068, 14037869, 14051911, 14065110, 14077865, 14089659,
#                                  14100926, 14111235, 14120986, 14130035, 14137956, 14145166,
#                                  14151113, 14156496, 14160752, 14165641, 14168070, 14169666,
#                                  14170527, 14170937, 14171171, 14177592],
#                                 [3, 6, 6, 6, 6, 7,
#                                  9, 11, 12, 14, 16, 16,
#                                  17, 18, 21, 22 ,27, 30,
#                                  32, 36, 41, 51, 58, 66,
#                                  76, 91, 103, 119, 134, 153,
#                                  177, 211, 258, 283, 324, 357,
#                                  423, 493, 557, 637, 765, 875,
#                                  1011, 1156, 1322, 1527, 1747, 1964,
#                                  2262, 2588, 2965, 3404, 3921, 4517,
#                                  5199, 5976, 6926, 7970, 9156, 10496,
#                                  12046, 13774, 15812, 18173, 20914, 24016,
#                                  27491, 31555, 35998, 41221, 47029, 53493,
#                                  61085, 69517, 78884, 89351, 101313, 114233,
#                                  129002, 145448, 163467, 183584, 205795, 230283,
#                                  257142, 287185, 320152, 356435, 395153, 438538,
#                                  485120, 536667, 592400, 653054, 720149, 791416,
#                                  870140, 954841, 1046164, 1145727, 1253317, 1370332,
#                                  1497058, 1635537, 1785475, 1949092, 2129151, 2328170,
#                                  2550322, 2798972, 3080212, 3404930, 3784826, 4236166,
#                                  4787602, 5492567, 6455252, 7688885, 8478058, 9141464,
#                                  9713122, 10209945, 10645267, 11026994, 11359936, 11652528,
#                                  11909166, 12135126, 12334971, 12511964, 12669079, 12808658,
#                                  12934062, 13045990, 13145318, 13234532, 13314516, 13386861,
#                                  13451743, 13510826, 13564219, 13612411, 13656753, 13696730,
#                                  13732973, 13766034, 13796471, 13824385, 13849911, 13873438,
#                                  13895194, 13914999, 13933317, 13950317, 13965683, 13979947,
#                                  13993384, 14005739, 14017117, 14027734, 14037614, 14046727,
#                                  14055305, 14063241, 14070735, 14077773, 14084415, 14090592,
#                                  14096408, 14101729, 14106791, 14111578, 14116076, 14120289,
#                                  14124218, 14127918, 14131288, 14134331, 14137242, 14139998,
#                                  14142696, 14145252, 14147639, 14149878, 14151972, 14153950,
#                                  14155763, 14157482, 14159068, 14160596, 14161954, 14163267,
#                                  14164540, 14165752, 14166832, 14167837, 14168798, 14169666,
#                                  14170456, 14171183, 14171932, 14172520, 14173118, 14173656,
#                                  14174137, 14174510, 14174894, 14175260, 14175591, 14175866,
#                                  14176096, 14176328, 14176498, 14176679, 14176825, 14176955,
#                                  14177053, 14177140, 14177229, 14177299, 14177363, 14177408,
#                                  14177445, 14177474, 14177489, 14177504, 14177524, 14177530,
#                                  14177542, 14177554, 14177563, 14177565, 14177569, 14177578,
#                                  14177578, 14177578, 14177580, 14177581, 14177582, 14177584,
#                                  14177585, 14177585, 14177587, 14177588, 14177588, 14177588,
#                                  14177589, 14177589, 14177590, 14177590, 14177590, 14177590,
#                                  14177590, 14177590, 14177590, 14177592],
#                                 [1, 1, 2, 2, 3, 4,
#                                  4, 7, 15, 20, 21, 33,
#                                  36, 42, 44, 51, 54, 66,
#                                  75, 83, 92, 113, 124, 136,
#                                  152, 172, 194, 221, 244, 269,
#                                  304, 345, 386, 433, 471, 513,
#                                  557, 614, 675, 719, 798, 867,
#                                  957, 1045, 1117, 1218, 1309, 1406,
#                                  1542, 1684, 1833, 2007, 2190, 2375,
#                                  2578, 2819, 3033, 3302, 3552, 3842,
#                                  4157, 4517, 4863, 5300, 5746, 6254,
#                                  6794, 7347, 7951, 8604, 9297, 10120,
#                                  11017, 11892, 12917, 13984, 15090, 16361,
#                                  17663, 19203, 20755, 22405, 24272, 26323,
#                                  28499, 30965, 33517, 36370, 39403, 42849,
#                                  46487, 50305, 54434, 58936, 63811, 69051,
#                                  74758, 81059, 87804, 95073, 103044, 111755,
#                                  121154, 131301, 142279, 154075, 167089, 180961,
#                                  195963, 212447, 229984, 249179, 270095, 292347,
#                                  316524, 343035, 371605, 402743, 436388, 473627,
#                                  514094, 558202, 606157, 659539, 717221, 781158,
#                                  852264, 930680, 1017975, 1116464, 1228358, 1356660,
#                                  1506784, 1684989, 1903362, 2180025, 2549013, 3290340,
#                                  3811840, 4303470, 4772340, 5225323, 5661862, 6084222,
#                                  6491760, 6884950, 7263323, 7627834, 7979711, 8316995,
#                                  8640259, 8950119, 9245564, 9529069, 9798366, 10056803,
#                                  10302265, 10536112, 10756876, 10968042, 11168289, 11358039,
#                                  11537336, 11708550, 11868980, 12020835, 12164599, 12300002,
#                                  12427839, 12549320, 12662702, 12769681, 12870029, 12964537,
#                                  13053466, 13136652, 13215123, 13288279, 13356952, 13420449,
#                                  13480576, 13535842, 13588023, 13636185, 13681415, 13723144,
#                                  13762193, 13798156, 13831410, 13862252, 13890598, 13916979,
#                                  13941334, 13963503, 13983926, 14002354, 14019472, 14035233,
#                                  14049387, 14062109, 14074069, 14085069, 14094901, 14103836,
#                                  14112034, 14119228, 14125631, 14131549, 14136722, 14141364,
#                                  14145592, 14149398, 14152694, 14155767, 14158480, 14160950,
#                                  14163054, 14164943, 14166594, 14168053, 14169369, 14170467,
#                                  14171350, 14172249, 14173036, 14173742, 14174314, 14174786,
#                                  14175213, 14175616, 14175943, 14176222, 14176473, 14176688,
#                                  14176863, 14177011, 14177124, 14177225, 14177300, 14177364,
#                                  14177409, 14177447, 14177472, 14177500, 14177523, 14177537,
#                                  14177554, 14177565, 14177575, 14177582, 14177584, 14177586,
#                                  14177590, 14177590, 14177591, 14177592]]],
#                               dtype=self.dtype, device=self.device, requires_grad=False)
#         return labsum
