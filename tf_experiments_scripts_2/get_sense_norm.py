from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

import my_tf_pkg as mtf

def get_pairwise_norm_squared(X,Y):
    return mtf.euclidean_distances(X=X,Y=Y,squared=True)

# tasks
#task_name = 'hrushikesh'
task_name = 'task_MNIST_flat_auto_encoder'
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mtf.get_data(task_name)

print 'X_train.shape', X_train.shape
print 'X_cv.shape', X_cv.shape
print 'X_test.shape', X_test.shape

pairwise_norm_squared = get_pairwise_norm_squared(X=X_train,Y=X_train)
min_val, max_val, mean_val, std_val = float(np.amin(pairwise_norm_squared)), float(np.amax(pairwise_norm_squared)), float(np.mean(pairwise_norm_squared)), float(np.std(pairwise_norm_squared))
print 'min norm X **2: ', np.amin(pairwise_norm_squared)
print 'max norm X **2: ', np.amax(pairwise_norm_squared)
print 'mean norm X **2: ', np.mean(pairwise_norm_squared)
print 'std norm X **2: ', np.std(pairwise_norm_squared)

# mtf.make_and_check_dir(path=tensorboard_data_dump_train)
# with open('./get_sense_norm'+, 'w+') as f_json:
#     json.dump(results,f_json,sort_keys=True, indent=2, separators=(',', ': '))
