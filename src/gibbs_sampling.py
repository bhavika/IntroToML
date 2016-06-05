import random
from collections import defaultdict

def roll_a_die():
    return random.choice([1,2,3,4,5,6])


def direct_sample():
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1+d2


def random_y_given_x(x):
    """equally likely to be x+ 1, x+2 , .... , x+6"""
    return x + roll_a_die()


def random_x_given_y(y):
    if y <= 7:
        # if the total is 7 or less, the first die is equally likely to be 1, 2, .... (total-1)
        return random.randrange(1, y)

    else:
        # if the total is 7 or more, the first die is equally likely to be (total-6) , (total-5),...6
        return random.randrange(y - 6, 7)


def gibbs_sample(num_iters=100):
    x, y = 1, 2
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y


def compare_distributions(num_samples=1000):
    counts = defaultdict(lambda : [0 , 0])
    for _ in range(num_samples):
        counts[gibbs_sample()][0] += 1
        counts[gibbs_sample()][1] += 1
    return counts

