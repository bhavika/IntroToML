import  math


def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sigmoid(t):
        return 1 / (1 + math.exp(-t))