import numpy as np
import unittest

def tri(a, m, s, x, t):
    b = 1.0 / s**2.0
    return a * tf.maximum(0.0, m - b * l1_norm(x, t)) 

def l1_norm(x, t):
    n, d = x.shape
    return tf.norm(x.reshape((n, 1, d)) - t, ord=1, axis=2)

class TriTest(unittest.TestCase):
    sess = tf.InteractiveSession()

    def test_norm_one_point_one_center(self):
        x = np.array([[1, 2, 3]]).astype(np.float32)
        t = np.array([[4, 3, 1]]).astype(np.float32)
        norm = l1_norm(x, t).eval()
        exp = 6
        assert_array_equal(norm, exp)

    def test_norm_one_point_three_centers(self):
        x = np.array([[1, 2, 3]]).astype(np.float32)
        t = np.array([[4, 3, 1], [2, 5, 9], [9, 3, 1]]).astype(np.float32)
        norm = l1_norm(x, t).eval()
        exp = np.array([[6, 10, 11]]).astype(np.float32)
        assert_array_equal(norm, exp)

    def test_norm_two_points_one_center(self):
        x = np.array([[5, 1, 2], [3, 1, 7]]).astype(np.float32)
        t = np.array([[1, 3, 1]]).astype(np.float32)
        norm = l1_norm(x, t).eval()
        exp = np.array([[7],[10]]).astype(np.float32)
        assert_array_equal(norm, exp)

    def test_norm_three_points_two_centers(self):
        x = np.array([[1, 2, 3],[2, 5, 9], [9, 3, 1]]).astype(np.float32)
        t = np.array([[4, 3, 1],[5, 6, 2]]).astype(np.float32)
        norm = l1_norm(x, t).eval()
        exp = np.array([[6, 9], [12, 11], [5, 8]]).astype(np.float32)
        assert_array_equal(norm, exp)

    def test_tri_1d_multiple_points_centers(self):
        a = 2; m = 1; s = 2
        x = np.array([[3.5],[0.2]]).astype(np.float32)
        t = np.array([[0],[3]]).astype(np.float32)
        print (tri(a, m, s, x, t).eval())
