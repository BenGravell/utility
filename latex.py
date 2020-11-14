def print_latex_matrix(A, places=None):
    n = A.shape[0]
    m = A.shape[1]
    print('\\begin{bmatrix}')
    for i, row in enumerate(A):
        line = '    '
        for j, entry in enumerate(row):
            if places is None:
                entry_str = str(entry)
            else:
                entry_str = '%.*f' % (places, entry)
            line += entry_str
            if j < m-1:
                line += ' & '
        if i < n-1:
            line += ' \\\\'
        print(line)
    print('\\end{bmatrix}')
    print('')


if __name__ == "__main__":
    import numpy as np
    print_latex_matrix(np.random.rand(3, 3).round(3), places=3)
