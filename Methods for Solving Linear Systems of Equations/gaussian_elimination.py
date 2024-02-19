import numpy as np
from colors import bcolors
from matrix_utility import swap_row
from condition_of_linear_equations import condition_number


def gaussianElimination(mat):
    N = len(mat)

    singular_flag = forward_substitution(mat)

    if singular_flag != -1:

        if mat[singular_flag][N]:
            return "Singular Matrix (Inconsistent System)"
        else:
            return "Singular Matrix (May have infinitely many solutions)"

    # if matrix is non-singular: get solution to system using backward substitution
    return backward_substitution(mat)




def forward_substitution(mat):
    N = len(mat)
    for k in range(N):

        # Partial Pivoting: Find the pivot row with the largest absolute value in the current column
        pivot_row = k
        v_max = mat[pivot_row][k]
        for i in range(k + 1, N):
            if abs(mat[i][k]) > v_max:
                v_max = mat[i][k]
                pivot_row = i
        # if a principal diagonal element is zero,it denotes that matrix is singular,
        if abs(mat[k][k]) < 1e-10:
            mat[k][k] = 0
        # and will lead to a division-by-zero later.
        if not mat[k][pivot_row]:
            return k  # Matrix is singular

        # Swap the current row with the pivot row
        if pivot_row != k:
            swap_row(mat, k, pivot_row)
        # End Partial Pivoting

        for i in range(k + 1, N):

            #  Compute the multiplier
            m = mat[i][k] / mat[k][k]

            # subtract fth multiple of corresponding kth row element
            for j in range(k + 1, N + 1):
                mat[i][j] -= mat[k][j] * m

            # filling lower triangular matrix with zeros
            mat[i][k] = 0

    return -1


# function to calculate the values of the unknowns
def backward_substitution(mat):
    N = len(mat)
    x = np.zeros(N)  # An array to store solution

    # Start calculating from last equation up to the first
    for i in range(N - 1, -1, -1):

        x[i] = mat[i][N]

        # Initialize j to i+1 since matrix is upper triangular
        for j in range(i + 1, N):
            x[i] -= mat[i][j] * x[j]

        x[i] = (x[i] / mat[i][i])

    return x


if __name__ == '__main__':

    A_b = ([[1, 2, 3, 4, 5],
             [2, 3, 4, 5, 1],
             [0, 0, 0, 0, 0],
                  [24, 15, 22, 1, 8]])
    A_bs = np.array([[1, 2, 3, 4],
            [2, 3, 4, 5],
            [0, 3, 0, 0],
            [24, 15, 22, 1]])
    cond = condition_number(A_bs)
    if condition_number(A_bs) > 2:
        print(f"pay attention that the solution may not be exact, the cond: {cond}.\n ")


    result = gaussianElimination(A_b)
    if isinstance(result, str):
        print(result)
    else:
        print(bcolors.OKBLUE,"\nSolution for the system:")
        for x in result:
            print("{:.6f}".format(x))