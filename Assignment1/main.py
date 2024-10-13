import math
import re


# EX 1
def parse_equation(equation):
    equation = equation.replace(" ", "")
    x_coefficient, y_coefficient, z_coefficient = 0.0, 0.0, 0.0

    x_match = re.search(r'([+-]?\d*\.?\d*)x', equation)
    y_match = re.search(r'([+-]?\d*\.?\d*)y', equation)
    z_match = re.search(r'([+-]?\d*\.?\d*)z', equation)
    constant_match = re.search(r'=\s*([+-]?\d*\.?\d+)', equation)

    def parse_coefficient(match):
        if not match or match.group(1) == '':
            return 1.0
        elif match.group(1) == '+' or match.group(1) == '-':
            return float(match.group(1) + '1')
        else:
            return float(match.group(1))

    if x_match:
        x_coefficient = parse_coefficient(x_match)
    if y_match:
        y_coefficient = parse_coefficient(y_match)
    if z_match:
        z_coefficient = parse_coefficient(z_match)

    if not constant_match:
        raise ValueError(f"Cannot parse equation: {equation}")
    constant = float(constant_match.group(1))

    return [x_coefficient, y_coefficient, z_coefficient], constant


def read_system_of_equations(filename):
    matrix = []
    vector = []

    with open(filename, 'r') as file:
        for line in file:
            coefficients, constant = parse_equation(line.strip())
            matrix.append(coefficients)
            vector.append(constant)

    return matrix, vector


def determinant(matrix):
    if len(matrix) == 2 and len(matrix[0]) == 2:
        a11, a12 = matrix[0]
        a21, a22 = matrix[1]
        return a11 * a22 - a12 * a21


    elif len(matrix) == 3:
        return (
                matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
                matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
                matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )

    else:
        raise ValueError("Determinant can only be calculated for 2x2 or 3x3 matrices.")


def vector_norm(vector):
    return math.sqrt(sum(b**2 for b in vector))


def trace(matrix):
    if len(matrix) != 3 or len(matrix[0]) != 3:
        raise ValueError("Trace can only be calculated for a 3x3 matrix.")

    return matrix[0][0] + matrix[1][1] + matrix[2][2]


def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


def matrix_vector_multiplication(matrix, vector):
    if len(matrix[0]) != len(vector):
        raise ValueError("Matrix A columns must equal vector B length for multiplication.")

    result = []
    for i in range(len(matrix)):
        result.append(sum(matrix[i][j] * vector[j] for j in range(len(vector))))

    return result


def replace_column(matrix, col_index, vector):
    matrix_new = [row[:] for row in matrix]
    for i in range(len(B)):
        matrix_new[i][col_index] = vector[i]
    return matrix_new


def solve_cramer_rule(matrix, vector, det_matrix):

    if det_matrix == 0:
        raise ValueError("The system has no unique solution because determinant(A) is 0.")

    matrix_x = replace_column(matrix, 0, vector)
    matrix_y = replace_column(matrix, 1, vector)
    matrix_z = replace_column(matrix, 2, vector)

    det_matrix_x = determinant(matrix_x)
    det_matrix_y = determinant(matrix_y)
    det_matrix_z = determinant(matrix_z)

    res_x: float = det_matrix_x / det_matrix
    res_y: float = det_matrix_y / det_matrix
    res_z: float = det_matrix_z / det_matrix

    return res_x, res_y, res_z


def minor_matrix(matrix, i, j):
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]


def cofactor_matrix(matrix):
    cofactors = []
    for i in range(3):
        cofactor_row = []
        for j in range(3):
            minor = minor_matrix(matrix, i, j)
            cofactor = ((-1) ** (i + j)) * determinant(minor)
            cofactor_row.append(cofactor)
        cofactors.append(cofactor_row)
    return cofactors


def adjugate(matrix):
    cofactors = cofactor_matrix(matrix)
    return [[cofactors[j][i] for j in range(3)] for i in range(3)]


def inverse_matrix(matrix):
    det_matrix = determinant(matrix)
    if det_matrix == 0:
        raise ValueError("The matrix is singular and cannot be inverted.")

    adj = adjugate(matrix)
    # Multiply each element of adjugate matrix by 1 / determinant
    return [[adj[i][j] / det_matrix for j in range(3)] for i in range(3)]


def matrix_vector_multiply(matrix, vector):
    return [sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3)]


def solve_by_matrix_inversion(matrix, vector):
    inv_matrix = inverse_matrix(matrix)
    return matrix_vector_multiply(inv_matrix, vector)


A, B = read_system_of_equations('input.txt')
print("Matrix A:", A)
print("Vector B:", B)
det_A = determinant(A)
trace_A = trace(A)
norm_B = vector_norm(B)
transposed_A = transpose(A)
product_AB = matrix_vector_multiplication(A, B)
x, y, z = solve_cramer_rule(A, B, det_A)
X = solve_by_matrix_inversion(A, B)

print("Determinant of A:", det_A)
print("Trace of A:", trace_A)
print("Euclidean norm of B:", norm_B)
print("Transpose of A:", transposed_A)
print("Matrix A multiplied by Vector B:", product_AB)
print(f"Solution using Cramer: x = {x}, y = {y}, z = {z}")
print(f"Solution using Inversion: x = {X[0]}, y = {X[1]}, z = {X[2]}")
