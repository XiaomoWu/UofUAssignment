def f(x):
    assert x in [1, 2] or [3, 6], 'not in range!'
    return x

f(3)