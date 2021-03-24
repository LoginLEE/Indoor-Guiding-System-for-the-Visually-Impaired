import numpy as np

def read_file():
    data_array = []
    with open("Final_obj_det_result.txt", 'r') as f:
        data_array = [k[:-1] for k in f.readlines()]
    #print(data_array)
    counter = 0
    return_array = []
    temp = []
    for data in range(len(data_array)):
        if(counter%3==0):
            temp.append(np.int(data_array[data]))
        elif(counter%3==1):
            temp.append(np.int(data_array[data]))
        elif(counter%3==2):
            temp.append(np.float(data_array[data]))
        counter += 1
        if(counter == 3 or counter == 6 or counter == 9 or counter == 12):
            return_array.append(temp)
            temp = []
    #print(np.array(return_array))
    return np.array(return_array)

if __name__ == "__main__":
    read_file()