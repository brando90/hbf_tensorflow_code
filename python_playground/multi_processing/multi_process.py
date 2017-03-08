from multiprocessing import Pool
import contextlib
def my_model((param1, param2, param3)): # Note the extra (), required by the pool syntax
    < your code >

num_pool_worker=1 # can be bigger than 1, to enable parallel execution
with contextlib.closing(Pool(num_pool_workers)) as po: # This ensures that the processes get closed once they are done
     pool_results = po.map_async(my_model,
                                    ((param1, param2, param3)
                                     for param1, param2, param3 in params_list))
     results_list = pool_results.get()
