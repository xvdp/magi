"""@xvdp tet for magi/features/list_util.py"""
import random
from magi.features import list_subset, list_modulo, list_transpose


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
    subset = random.randint(0, len(in_list)-1)
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
