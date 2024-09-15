# Danial Chitnis, June 2020

import os
import pandas as pd
import subprocess

databaseFolder = './'
dirs = os.listdir(databaseFolder)

df = pd.DataFrame(columns=["index"])
df['mtx'] = ""
df['time1'] = 0
df['time1'] = df['time1'].astype(float)
df['time2'] = 0
df['time2'] = df['time1'].astype(float)


engine = './klu_threaded.o'

timeout = 100

###################### specs inclusive ################
#mtx_list = ['rajat11', 'circuit_2', 'circuit_4', 'hcircuit', 'scircuit', 'ASIC_680ks']
mtx_list = ['scircuit']

#runtimes = [200, 20, 10, 10, 10, 10]
runtimes = [10]

thread = 4
nrhs = 128
#######################################################


def runSingle(engine, threads, nrhs, mtx, runtime):
    try:
        filename = './'+mtx+'/'+mtx+'.mtx'
        # print(filename)
        bmatrix = './'+mtx+'/vecb.mtx'
        # print(bmatrix)
        command = [engine, str(threads), str(
            nrhs), filename, bmatrix, str(runtime)]
        # print(command)
        p = subprocess.run(command, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout)
        # print(p.stdout.decode('utf-8'))
        if p.returncode == 0:
            outLines = (p.stdout).decode('utf-8').split('\n')
            # print(outLines)
            time1 = float(outLines[-2].split(':')[1])

        else:
            print(p.stderr.decode('utf-8'))
            time1 = -1

    except subprocess.TimeoutExpired:
        time1 = -2

    return time1


#t1 = runSingle(engine, 1, 1)

loc = 0

for i in range(len(mtx_list)):
    t1 = runSingle(engine, 1, 1, mtx_list[i], runtimes[i])
    t2 = runSingle(engine, thread, nrhs, mtx_list[i], runtimes[i])
    df.loc[loc, 'index'] = loc
    df.loc[loc, 'mtx'] = mtx_list[i]
    df.loc[loc, 'time1'] = t1
    df.loc[loc, 'time2'] = t2
    print(mtx_list[i], t1, t2)
    loc += 1


print('')
print(df)
print('')

df.to_csv('results.csv')


print('done!')
print('')
