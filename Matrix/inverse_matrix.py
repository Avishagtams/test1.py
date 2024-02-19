from colors import bcolors
from matrix_utility import row_addition_elementary_matrix, scalar_multiplication_elementary_matrix, print_matrix
import numpy as np



"""
Function that find the inverse of non-singular matrix
The function performs elementary row operations to transform it into the identity matrix. 
The resulting identity matrix will be the inverse of the input matrix if it is non-singular.
 If the input matrix is singular (i.e., its diagonal elements become zero during row operations), it raises an error.
"""


def inverse(matrix):
    print(bcolors.OKBLUE, f"=================== Finding the 3 elementry first: ===================\n {matrix}\n", bcolors.ENDC)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]
    identity = np.identity(n)
    counter = 3

    # Perform row operations to transform the input matrix into the identity matrix
    for i in range(n):
        if matrix[i, i] == 0 :
            raise ValueError("Matrix is singular, cannot find its inverse.")

        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(n, i, scalar)
            if counter > 0:
                print(f"elementary matrix number: {4-counter} to make the diagonal element 1 :\n {elementary_matrix} \n")
                counter -= 1
                matrix = np.dot(elementary_matrix, matrix)
                #print(f"The matrix after elementary operation :\n {matrix}")
                #print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",  bcolors.ENDC)
                identity = np.dot(elementary_matrix, identity)


        # Zero out the elements above and below the diagonal
        for j in range(n):
            if i != j:
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                if counter > 0:
                    print(f"elementary matrix number: {4-counter} for R{j+1} = R{j+1} + ({scalar}R{i+1}):\n {elementary_matrix} \n")
                    counter -= 1
                    matrix = np.dot(elementary_matrix, matrix)
                    #print(f"The matrix after elementary operation :\n {matrix}")
                    #print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",bcolors.ENDC)
                    identity = np.dot(elementary_matrix, identity)

    return identity
"""
find norm:
"""
def norm(mat):
    size = len(mat)
    max_row = 0
    for row in range(size):
        sum_row = 0
        for col in range(size):
            sum_row += abs(mat[row][col])
        if sum_row > max_row:
            max_row = sum_row
    return max_row

def normMax (A):
    norm_A = norm(A)
    return norm_A



if __name__ == '__main__':
    """"
    Date: 19/2/24
    Group: Avishag Tamssut id-326275609
            Merav Hashta id-214718405
            Sahar Emmuna id-213431133
    Git: https://github.com/Avishagtams/test1.py.git
    Name: Avishag Tamssut 326275609
    """
    A = np.array([[2, 1, 0],
                  [3, -1, 0],
                  [1, 4, -2]])
    try:
        A_inverse = inverse(A)
        #print(bcolors.OKBLUE, "\nInverse of matrix A: \n", A_inverse)
        print("=====================================================================================================================", bcolors.ENDC)

    except ValueError as e:
        print(str(e))

maxNormal = normMax(A)
print(f"The max normal is: {maxNormal}.\n")