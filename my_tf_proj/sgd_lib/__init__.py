# helps users of project/pkg from knowning the internal structure of modules
# easier to use funcs in all modules
# following line imports (i.e. similar to copying the code) from the declared packages
from sgd_lib.main_basin import *
from sgd_lib.GDL import *
#having the package name declared
#from pkg_1.module2 import *

#from pkg_1.module1 import f1 as superduperf1
# 2 options to import
# (1) from pkg_1.module1 import f1
# (2) from pkg_1 import superduperf1