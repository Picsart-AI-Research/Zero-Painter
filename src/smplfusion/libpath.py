import os
from os.path import *
import sys

import pickle
import inspect
import json
import importlib.util
import importlib
import random
import csv

# Requirements
import yaml
from omegaconf import OmegaConf
import torch
import numpy as np
from PIL import Image
from IPython.display import Markdown  # TODO: Remove this requirement

from .libimage import IImage
# from .experiment import SmartFolder

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def load_obj(objyaml):
    if "__init__" in objyaml:
        return get_obj_from_str(objyaml['__class__'])(**objyaml["__init__"])
    else:
        return get_obj_from_str(objyaml['__class__'])()

def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    try:
        return getattr(importlib.import_module(module, package=None), cls)
    except:
        return getattr(importlib.import_module('lib.' + module, package=None), cls)


def path2var(path):
    return path.replace(" ", "_").replace(".", "_").replace("-", "_").replace(":", "_")


def var2path(path):
    return path.replace("ß", " ").replace("ä", ".").replace("ö", "-")


def save(obj, path):
    pass

class FileType:
    pass

def cvt(x: str):
    try: return int(x)
    except: 
        try: return float(x)
        except: return x

class Path:
    FOLDER = FileType()

    def cwd(path):
        return Path(os.path.dirname(os.path.abspath(path)))
    def __init__(self, path):
        self.path = path
        if os.path.isdir(self.path):
            self.reload()
        else:
            self.extension = self.path.split(".")[-1]
            self.subpaths = {}
        self.filename = os.path.basename(self.path)

    def reload(self):
        if os.path.isdir(self.path):
            self.subpaths = {path2var(x): x for x in os.listdir(self.path)}
            self.keyorder = sorted(list(self.subpaths.keys()))
            self.extension = None

    def read(self, is_preview=False):
        if self.extension is None or is_preview:
            # if self.isdir and os.path.exists(f'{self}/__smart__'): 
            #     return SmartFolder.open(str(self))
            return self
        elif self.extension == "yaml":
            yaml = OmegaConf.load(self.path)
            if "__class__" in yaml:
                return load_obj(yaml)
            if "model" in yaml:
                return instantiate_from_config(yaml.model)
            return yaml
        elif self.extension == "yamlobj":
            return load_obj(OmegaConf.load(self.path))
        elif self.extension in ["jpg", "jpeg", "png"]:
            return IImage.open(self.path)
        elif self.extension in ["ckpt", "pt"]:
            return torch.load(self.path)
        elif self.extension in ["pkl", "bin"]:
            with open(self.path, "rb") as f:
                return pickle.load(f)
        elif self.extension in ["txt"]:
            with open(self.path) as f:
                return f.read()
        elif self.extension in ["lst", "list"]:
            with open(self.path) as f:
                return [cvt(x) for x in f.read().split("\n")]
        elif self.extension in ["md"]:
            with open(self.path) as f:
                return Markdown(f.read())
        elif self.extension in ["json"]:
            with open(self.path) as f:
                return json.load(f)
        elif self.extension in ["npy", "npz"]:
            return np.load(self.path)
        elif self.extension in ['csv', 'table']:
            with open(self.path) as f:
                return [[cvt(x) for x in row] for row in csv.reader(f, delimiter=' ')]
        elif self.extension in ["py"]:
            spec = importlib.util.spec_from_file_location(f"autoload.module", self.path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        else:
            return self

    def exists(self):
        return os.path.exists(str(self))

    def sample(self):
        return getattr(self, random.choice(list(self.subpaths.keys())))

    def __dir__(self):
        self.reload()
        return list(self.subpaths.keys())

    def __len__(self):
        return len(self.subpaths)

    def ls(self):
        self.reload()
        return list(self.subpaths.values())
        # return [str(self + x) for x in self.subpaths.values()]
        # return dir(self)

    @property
    def parent(self): return Path(os.path.dirname(self.path))
    @property
    def isdir(self): return os.path.isdir(os.path.abspath(self.path))

    def __getattr__(self, __name: str):
        # print ("Func", inspect.stack()[1].function, "name", __name)
        is_preview = inspect.stack()[1].function == "getattr_paths"

        self.reload()
        if __name in self.subpaths and self.subpaths[__name] in os.listdir(self.path):
            return Path(join(self.path, self.subpaths[__name])).read(is_preview)
        elif __name in ['__wrapped__']:
            raise AttributeError()
        else:
            return Path(join(self.path, __name))
        
    def __setattr__(self, __name: str, __value: any):
        if __value == Path.FOLDER:
            os.makedirs(f'{self}/{__name}', exist_ok=True)
            self.reload()
        else:
            return super().__setattr__(__name, __value)
    
    def __hasattr__(self, __name: str):
        self.reload()
        return self.subpaths is not None and __name in self.subpaths

    # def __getitem__(self, __name):
    #     if __name in self.subpaths and self.subpaths[__name] in os.listdir(self.path):
    #         return Path(join(self.path, self.subpaths[__name])).read()

    def __add__(self, other):
        assert other is not str
        return Path(join(self.path, other))

    def __call__(self, idx=None):
        if idx is None:
            return str(self)
        if isinstance(idx, str):
            return Path(join(self.path, idx))
        else:
            return Path(join(self.path, self.subpaths[self.keyorder[idx]]))

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return Path(join(self.path, idx)).read()
        else:
            return Path(join(self.path, self.subpaths[self.keyorder[idx]])).read()

    def __str__(self):
        return os.path.abspath(self.path)

    def __repr__(self):
        return f"Path reference to: {os.path.abspath(self.path)}"

    def new_child(self, ext = ''):
        idx = 0
        while os.path.exists(join(str(self), f"file_{idx}{ext}")):
            idx += 1
        return join(str(self), f"file_{idx}{ext}")