

import importlib
import sys

# ---------------------------------------------------------------

def import_from_file(name, loc, reload = False):
    spec = importlib.util.spec_from_file_location(name, loc)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    
# ---------------------------------------------------------------
