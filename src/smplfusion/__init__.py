# TODO: Remove farancia.Path depency!
from .libpath import Path
from .libimage import IImage

print (Path.cwd(__file__))
config = Path.cwd(__file__).config
models = Path.cwd(__file__).config
options = Path.cwd(__file__).config