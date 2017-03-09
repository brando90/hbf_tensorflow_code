from multiprocessing import Pool
import contextlib
import time

def my_code(x):
    time.sleep(2)
    xx = x*x
    print('x*x: ',xx)
    return xx

num_pool_workers=1 # can be bigger than 1, to enable parallel execution
with contextlib.closing( Pool(num_pool_workers) ) as po: # This ensures that the processes get closed once they are done
    print('start')
    pool_results = po.map_async( my_code, [1,2,3] )
    results_list = pool_results.get()
    print('results_list: ', results_list)
    print('Done')
