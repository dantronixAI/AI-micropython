from .numpy_package import numpy as np
import os
path =__file__
path = path.replace("storage.py","")

def save_layer(layer,n_layer,ltype):
    f = open(path+"output/{}".format(ltype)+str(n_layer), "w")
    shape = layer.shape
    f.write("Layer {}: {}".format(n_layer,layer.shape))
    f.write("\n")
    for row in range(shape[0]):
        for col in range(shape[1]):
            f.write("{}".format(layer[row][col],))
            if not ((row==shape[0]-1) and (col==shape[1]-1)):
                f.write(",")
    f.close()

def load_layer(n_layer,ltype):
    f = open(path + "output/{}".format(ltype) + str(n_layer), "r")
    line = f.readline()
    ini_pos = line.find("(")
    end_pos = line.find(")")
    structure = line[ini_pos + 1:end_pos]
    dimentions = structure.split(",")
    rows = int(dimentions[0])
    cols = int(dimentions[1])
    array = np.zeros((rows, cols))
    line = f.readline()
    line = line.replace("\n", "")
    line = line.split(",")
    c_line = 0
    for row in range(rows):
        for col in range(cols):
            array[row][col] = float(line[c_line])
            c_line += 1
    f.close()
    return array

def save_2darray(array,name):
    f = open(path+"output/{}".format(name), "w")
    shape = array.shape
    f.write("{}: {}".format(name,shape))
    f.write("\n")
    for row in range(shape[0]):
        for col in range(shape[1]):
            f.write("{}".format(array[row][col],))
            if not ((row==shape[0]-1) and (col==shape[1]-1)):
                f.write(",")
    f.close()

def load_2darray(name):
    f = open(path + "output/{}".format(name), "r")
    line = f.readline()
    ini_pos = line.find("(")
    end_pos = line.find(")")
    structure = line[ini_pos + 1:end_pos]
    dimentions = structure.split(",")
    rows = int(dimentions[0])
    cols = int(dimentions[1])
    array = np.zeros((rows, cols))
    line = f.readline()
    line = line.replace("\n", "")
    line = line.split(",")
    c_line = 0
    for row in range(rows):
        for col in range(cols):
            array[row][col] = float(line[c_line])
            c_line += 1
    f.close()
    return array


def save_weights(nn,name):
    data = nn.weights
    f = open("{}output/{}".format(path,name), "w")
    n_layers = len(data)
    f.write("# Layers:"+str(n_layers))
    f.write("\n")
    for k,layer in enumerate(data):
        shape = layer.shape
        f.write("Layer {}: {}".format(k,layer.shape))
        f.write("\n")
        for row in range(shape[0]):
            for col in range(shape[1]):
                f.write("{}".format(layer[row][col],))
                if not ((row==shape[0]-1) and (col==shape[1]-1)):
                    f.write(",")
        f.write("\n")
    f.close()

def restore_weights(nn,name):
    f = open("{}output/{}".format(path,name), "r")
    header = f.readline()
    data =[]
    lines = f.readlines()
    n_line = 0
    while n_line<len(lines):
        line = lines[n_line]
        if "Layer" in line:
            ini_pos = line.find("(")
            end_pos = line.find(")")
            structure = line[ini_pos+1:end_pos]
            dimentions = structure.split(",")
            rows = int(dimentions[0])
            cols = int(dimentions[1])
        n_line+=1
        array = np.zeros((rows,cols))
        line = lines[n_line].replace("\n","")
        line = line.split(",")
        c_line = 0
        for row in range(rows):
            for col in range(cols):
                array[row][col]=float(line[c_line])
                c_line+=1
        data.append(array)
        n_line += 1
    f.close()
    nn.weights=data
