# helps users of project/pkg from knowning the internal structure of modules
# easier to use funcs in all modules
# following line imports (i.e. similar to copying the code) from the declared packages
from my_tf_pkg.hbf_nn_builder import *
from my_tf_pkg.f_1D_data import *
from my_tf_pkg.initializations import *
from my_tf_pkg.my_rand_lib import *
from my_tf_pkg.process_sys_argv import *
from my_tf_pkg.krls_and_cv import *
from my_tf_pkg.save_workspace import *
from my_tf_pkg.f_2D_data import *
from my_tf_pkg.f_4D_data import *
from my_tf_pkg.f_8D_data import *
from my_tf_pkg.extract_results_lib2 import *
from my_tf_pkg.main_nn import *
from my_tf_pkg.f_4D_BT_data import *
from my_tf_pkg.general_bt_f import *
from my_tf_pkg.general_bt_NN import *
from my_tf_pkg.data_central_manager import *
#having the package name declared
#from pkg_1.module2 import *

#from pkg_1.module1 import f1 as superduperf1
# 2 options to import
# (1) from pkg_1.module1 import f1
# (2) from pkg_1 import superduperf1
