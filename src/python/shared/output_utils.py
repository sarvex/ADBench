# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

PRECISION = 8   # in signs after dot

def objective_file_name(output_prefix, input_basename, module_basename):
    return output_prefix + input_basename + "_F_" + module_basename + ".txt"

def jacobian_file_name(output_prefix, input_basename, module_basename):
    return output_prefix + input_basename + "_J_" + module_basename + ".txt"



def time_to_string(objective_time, derivative_time):
    obj_time_str = np.format_float_scientific(
        objective_time,
        unique=False,
        precision=PRECISION
    )

    der_time_str = np.format_float_scientific(
        derivative_time,
        unique=False,
        precision=PRECISION
    )

    return f"{obj_time_str}\n{der_time_str}"

def save_time_to_file(filepath, objective_time, derivative_time):
    with open(filepath,"w") as out:
        out.write(time_to_string(objective_time, derivative_time))
    


def value_to_string(value):
    return np.format_float_scientific(value, unique=False, precision=PRECISION)

def save_value_to_file(filepath, value):
    with open(filepath,"w") as out:
        out.write(value_to_string(value))

def save_vector_to_file(filepath, gradient):
    with open(filepath,"w") as out:
        for value in gradient:
            out.write(value_to_string(value) + '\n')

def save_jacobian_to_file(filepath, jacobian):
    with open(filepath,"w") as out:
        # output row-major matrix
        for row in jacobian:
            out.write(value_to_string(row[0]))
            for value in row[1:]:
                out.write('\t' + value_to_string(value))
            out.write('\n')

def save_errors_to_file(filepath, reprojection_error, zach_weight_error):
    with open(filepath,"w") as out:
        out.write("Reprojection error:\n")
        for value in reprojection_error:
            out.write(value_to_string(value) + '\n')

        out.write("Zach weight error:\n")
        for value in zach_weight_error:
            out.write(value_to_string(value) + '\n')

def save_sparse_j_to_file(filepath, J):
    with open(filepath,"w") as out:
        out.write(f"{J.nrows} {J.ncols}\n")

        out.write(f"{len(J.rows)}\n")
        for row in J.rows:
            out.write(f"{row} ")
        out.write('\n')

        out.write(f"{len(J.cols)}\n")
        for column in J.cols:
            out.write(f"{column} ")
        out.write('\n')

        for value in J.vals:
            out.write(f'{value_to_string(value)} ')