class Series(object):
    def __init__(self, low, high):
        self.current = low
        self.high = high

    def __iter__(self):
        return self

    def __next__(self):
        if self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            return self.current - 1

n_list = Series(1,10)    
print(list(n_list))

(i for i in [1, 2, 3])
[i for i in [1, 2, 3]]

def foo():
    result = []
    for i in [1, 2, 3]:
        x = i**2
        result.append(x)
    return result

def iterate_over():
    for i in [1, 2, 3]:
        x = i**2
        yield x

x = [1, 2, 3]
d = {'x': x}
x[0] = 999
d

def fib():
    prev, curr = 0, 1
    while True:
        yield curr
        prev, curr = curr, curr + prev
f = fib()
next(f)

from itertools import islice
def inf():
    start = 1
    while start <= 100:
        yield start
        start += 1
min(inf())

d = {'a':1, 'b': 0}
max(d.items(), key = lambda x: x[1])

d.values().sort()

def get_lines(f):
        for line in f:
            yield line

with open('data/train.liblinear') as f:
    print(type(f))
    first_line = next(f)

    for i, line in enumerate(f):
        print(i, line)
        if i > 10:
            break

x = iter([1, 2, 3])
type(x)

def gr(x):
    for i in x:
        yield i

type(gr(x))