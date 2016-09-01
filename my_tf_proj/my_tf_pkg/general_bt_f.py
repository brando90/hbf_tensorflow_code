import np

def f(x, h_list, l, left, right):
    if l == 1:
        h_l = h_list[l]
        return h_l(x[left,right])
    else:
        h = h_list[l]
        h_left, h_right = f(x, h_list, l, left, high/2), f(x, h_list, l, high/2, high)
        return h( [h_left,h_right])


def unit_test_BT4D(N_train=100, low_x=-1, high_x=1):
    h11 = lambda A: (2.0*A[0] + 3.0*A[1])**4.0
    h21 = lambda A: (4*A[0] + 5*A[1])**0.5
    h_list = [h11, h21]
    h = f_4D_conv_2nd
    # compare the functions
    X_train = low_x + (high_x - low_x) * np.random.rand(N_train,D)
    Y_train = get_labels_4D(X_train, f)
    return False

if __name__ == '__main__':
