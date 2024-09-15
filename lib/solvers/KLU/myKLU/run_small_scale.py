# Danial Chitnis, June 2020

import os
import pandas as pd
import subprocess

databaseFolder = './'
dirs = os.listdir(databaseFolder)

df = pd.DataFrame(columns=["index"])
df['threads'] = 0
df['nrhs'] = 0
df['time1'] = 0
df['time1'] = df['time1'].astype(float)
df['time2'] = 0
df['time2'] = df['time1'].astype(float)


engine = './klu_threaded.o'

timeout = 100
reps = 10

###################### specs inclusive ################
max_thread = 9  # 9
max_nrhs_p = 7  # 7  # 2^p
#######################################################


def runSingle(engine, threads, nrhs):
    try:
        #filename = './circuit_2/circuit_2.mtx'
        #bmatrix = './circuit_2/vecb.mtx'
        filename = './scircuit/scircuit.mtx'
        bmatrix = './scircuit/vecb.mtx'

        command = [engine, str(threads), str(
            nrhs), filename, bmatrix, str(reps)]
        # print(command)
        p = subprocess.run(command, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout)
        # print(p.stdout.decode('utf-8'))
        if p.returncode == 0:
            outLines = (p.stdout).decode('utf-8').split('\n')
            # print(outLines)
            time1 = float(outLines[-3].split(':')[1])
            time2 = float(outLines[-2].split(':')[1])

        else:
            print(p.stderr.decode('utf-8'))
            time1 = -1
            time2 = -1

    except subprocess.TimeoutExpired:
        time1 = -2
        time2 = -2

    return time1, time2


#t1 = runSingle(engine, 1, 1)

loc = 0

for threads in range(1, max_thread+1):
    for p in range(0, max_nrhs_p+1):
        nrhs = 2**p
        t1, t2 = runSingle(engine, threads, nrhs)
        df.loc[loc, 'index'] = loc
        df.loc[loc, 'threads'] = threads
        df.loc[loc, 'nrhs'] = nrhs
        df.loc[loc, 'time1'] = t1
        df.loc[loc, 'time2'] = t2
        print(threads, nrhs, t1, t2)
        loc += 1


print('')
print(df)
print('')

df.to_csv('results.csv')


print('done!')
print('')
