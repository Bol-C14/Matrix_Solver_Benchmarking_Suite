import scipy.io as sio

try:
    matrix = sio.mmread('/home/gushu/work/MLtask/ML_Circuit_Matrix_Analysis/data/ss_organized_data/494_bus.mtx')
    print(matrix)
except Exception as e:
    print(f"Error reading matrix: {e}")