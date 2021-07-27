#!/usr/bin/python2.7

def create_log(path, params):
    with open(path, 'w') as f:
        f.write('modelname= ' + params[0] + '\n')
        f.write('dataset= ' +params[1] +'\n')
        f.write('splits= ' +params[2] + '\n')
        f.write('layers= ' + params[3] + '\n')
        f.write('numstages= ' + params[4] + '\n')
        f.write('weighted= ' + params[5] + '\n')
        f.write('lambda= ' + params[6] + '\n')
        f.write('batchsize= ' + str(params[7]) + '\n')
        f.write('learningrate= ' + str(params[8]) + '\n')
        f.write('epochs= ' +str(params[9]) + '\n')
        f.write('\n')