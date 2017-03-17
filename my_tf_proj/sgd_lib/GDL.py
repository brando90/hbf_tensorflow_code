from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

def throw_error(grads_and_vars,loss):
    raise ValueError(
       "No gradients provided for any variable, check your graph for ops"
        " that do not support gradients, between variables %s and loss %s." %
        ([str(v) for _, v in grads_and_vars], loss))

class GDL(optimizer.Optimizer):

    def __init__(self, learning_rate, mu_noise=0.0, stddev_noise=1.0, use_locking=False, name="GDL"):
        """
        """
        super(GDL, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._mu_noise = mu_noise
        self._stddev_noise = stddev_noise


    def minimize(self, loss, global_step=None, var_list=None, gate_gradients=GATE_OP, aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None):
        '''
        comment
        '''
        # get gradients and variables
        grads_and_vars = self.compute_gradients( loss, var_list=var_list, gate_gradients=gate_gradients, aggregation_method=aggregation_method, colocate_gradients_with_ops=colocate_gradients_with_ops,  grad_loss=grad_loss)
        # get the trainable ones
        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            throw_error(grads_and_vars,loss)
        # do something to gradients
        processed_grads_and_vars = [ (_GDL(g),v) for (g,v) in grads_and_vars ]
        #apply the gradients
        return self.apply_gradients(processed_grads_and_vars, global_step=global_step, name=name)

    def _GDL(self,g):
        shape = tf.shape(g).eval()
        return g + tf.random_normal(shape,mean=self._mu_noise,stddev=self._stddev_noise)
