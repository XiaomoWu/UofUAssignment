from scipy.sparse import csr_matrix
a = csr_matrix([[1,2,0],[0,0,3],[4,0,5]])
rows, cols = a.nonzero()
for row, col in zip(rows, cols):
    a[row, col] = 1
a.todense()

data_train_compact = ld('data_train_compact')

@timeit('job')
def foo():
    x = data_train_compact.loc[0, 'x']
    #x = x.tolil()
    rows, cols = x.nonzero()
    for row, col in zip(rows, cols):
        x[row, col] = 1
foo()
x.todense()
