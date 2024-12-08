import objax
from objax import Module, ModuleList
import typing
from typing import List

def collect_modules(module: Module) -> List[Module]:
    """
    Recursively go every sub module and collect in collected_modules
    Will fail if loops exist.
    Does not remove duplicated.
    """
    collected_modules = []

    if isinstance(module, ModuleList):
        for idx, v in enumerate(module):
            if isinstance(v, Module):
                collected_modules.append(v) 

                collected_modules += collect_modules(v)
    else:
        for k, v in module.__dict__.items():
            if isinstance(v, ModuleList):
                # we want to flatten module lists
                collected_modules += collect_modules(v)
            elif isinstance(v, Module):
                collected_modules.append(v) 
                collected_modules += collect_modules(v)


    return collected_modules
