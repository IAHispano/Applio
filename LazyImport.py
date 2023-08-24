from importlib.util import find_spec, LazyLoader, module_from_spec
from sys import modules

def lazyload(name):
    if name in modules:
        return modules[name]
    else:
        spec = find_spec(name)
        loader = LazyLoader(spec.loader)
        module = module_from_spec(spec)
        modules[name] = module
        loader.exec_module(module)
        return module