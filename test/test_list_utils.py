"""@xvdp tet for magi/features/list_util.py"""
import random
import numpy as np
import torch
from magi.features import list_subset, list_modulo, list_transpose, list_intersect, list_flatten


_SENTENCE = "Die Welt ist durch die Tatsachen bestimmt und dadurch, dass es alle Tatsachen sind.".split(" ")
_abc = [['Die', 'Welt', 'ist', 'alles,', 'was', 'der', 'Fall', 'ist.'],
        ['Die', 'Welt', 'ist', 'die', 'Gesamtheit', 'der', 'Tatsachen,', 'nicht', 'der', 'Dinge.'],
        ['Die', 'Welt', 'ist', 'durch', 'die', 'Tatsachen', 'bestimmt', 'und', 'dadurch,', 'dass', 'es', 'alle', 'Tatsachen', 'sind.'],
        ['Denn,', 'die', 'Gesamtheit', 'der', 'Tatsachen', 'bestimmt,', 'was', 'der', 'Fall', 'ist', 'und', 'auch,', 'was', 'alles', 'nicht', 'der', 'Fall', 'ist.'],
        ['Die', 'Tatsachen', 'im', 'logischen', 'Raum', 'sind', 'die', 'Welt.'],
        ['Die', 'Welt', 'zerfällt', 'in', 'Tatsachen.'],
        ['Eines', 'kann', 'der', 'Fall', 'sein', 'oder', 'nicht', 'der', 'all', 'sein', 'und', 'alles', 'übrige', 'gleich', 'bleiben.']]

def test_list_subset_int():
    in_list = _SENTENCE
    subset = random.randint(1, len(in_list)-1)
    out_list = list_subset(in_list, subset)
    assert len(out_list) == subset

def test_list_subset_indices():
    in_list = _SENTENCE
    subset = list_modulo([-1, 0, 5, -2], len(in_list))
    out_list = list_subset(in_list, subset)
    assert out_list == ['sind.', 'Die', 'Tatsachen', 'Tatsachen']

def test_list_subset_text():
    in_list = _SENTENCE
    subset = ['bestimmt', 'durch', 'dadurc', 'Welt']
    out_list = list_subset(in_list, subset)
    assert len(subset) == len(out_list) + 1

def test_list_subset_text_notolerance():
    in_list = _SENTENCE
    subset = ['bestimmt', 'durch', 'dadurch,', 'Welt', 'lustig']
    out_list = list_subset(in_list, subset, tolerate=False)
    assert len(out_list) == len(in_list)

def test_list_transpose():
    a = [[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4]]
    assert list_transpose(a)[:3] == [1,1,1]

    assert list_transpose(a)[6:9] == [3,3,3]

def test_list_intersect():
    a = [1,2,3]
    b = [2,3,4]
    c = [3,4,5,6]
    out = list_intersect(a,b,c)
    assert out == [3]

# pylint: disable=no-member
torch.set_default_dtype(torch.float32) # reset float16 if set by previous test

def test_list_flatten():

    np_rg = np.arange(45,48,1, dtype="int")
    torch_rn = torch.ones(10, dtype=torch.int)*50 # pylint: disable=no-member
    nestlist = [[[10,11]]]
    tup = (9,3,6)
    out = list_flatten(np_rg, torch_rn, tup, nestlist, 1,2,3)
    assert len(out) == 21

    out = list_flatten(np_rg, torch_rn, tup, nestlist, 1,2,3, unique=True)
    assert len(out) == 11
