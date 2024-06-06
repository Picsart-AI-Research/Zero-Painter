from . import attentionpatch
from . import transformerpatch

VERBOSE = False
attention_forward = attentionpatch.default.forward
basic_transformer_forward = transformerpatch.default.forward

def reset():
   global attention_forward, basic_transformer_forward
   attention_forward = attentionpatch.default.forward
   basic_transformer_forward = transformerpatch.default.forward
   if VERBOSE: print ("Resetting Diffusion Model") 

print ("RELOADING ROUTER")
print (attention_forward)