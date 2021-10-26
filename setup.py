"""@xvdp
    setup file for magi module
"""
import os
import os.path as osp
import subprocess as sp
from setuptools import setup, find_packages

def _path(*args):
    return osp.join(osp.split(__file__)[0], *args)

def _readme():
    with open(_path('README.md')) as _fi:
        return _fi.read()

def _requirements():
    with open(_path('requirements.txt')) as _fi:
        return _fi.read().split()

def _set_version(version):
    with open(_path('magi','version.py'), 'w') as _fi:
        _fi.write('"""@xvdp generated by setup.py"""\n')
        _fi.write("__version__='"+version+"'\n")
    return version

def build_packages():
    """ setup """
    setup(
        name='magi',
        version=_set_version("0.0.5"),
        author="xvdp",
        url='https://github.com/xvdp/magi',
        license='MIT',
        description='magi, pytorch augments',
        long_description=_readme(),
        install_requires=_requirements(),
        packages=find_packages(),
        tests_require=["pytest"],
        include_package_data=True,
        python_requires='>=3.6',
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
    )

def build_submodules(submodules, root="extern"):
    """ hack / submodule install is weak"""
    _a = '\033[0m'
    _r = "'\033[91m\033[1m'"
    _dir = os.getcwd()
    for sub in submodules:
        submodule = osp.abspath(osp.join(osp.split(__file__)[0], root, sub))
        assert osp.isdir(submodule), f"{_r}Error: submodule file not found <{submodule}>: \n\t\t\twas project cloned with --recursive flag?{_a}"

        os.chdir(submodule)
        sp.run("python setup.py install", shell=True, check=True)
    os.chdir(_dir)


if __name__ == '__main__':
    build_packages()
    build_submodules(["koreto"], "extern")
