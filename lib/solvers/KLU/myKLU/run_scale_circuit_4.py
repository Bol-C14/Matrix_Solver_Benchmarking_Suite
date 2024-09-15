# Danial Chitnis, June 2020

import os
import pandas as pd
import subprocess

import run_scale_config as config

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

timeout = 20


def runSingle(engine, threads, nrhs, reps):
    try:
        filename = './circuit_4/circuit_4.mtx'
        bmatrix = './circuit_4/vecb.mtx'

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

for ti in range(0, len(config.threads_list)):
    for ni in range(0, len(config.nrhs_list)):
        thread = config.threads_list[ti]
        nrhs = config.nrhs_list[ni]

        if thread*nrhs <= 64:
            reps = 30
        else:
            reps = 10

        t1, t2 = runSingle(engine, thread, nrhs, reps)
        df.loc[loc, 'index'] = loc
        df.loc[loc, 'threads'] = thread
        df.loc[loc, 'nrhs'] = nrhs
        df.loc[loc, 'time1'] = t1
        df.loc[loc, 'time2'] = t2
        print(thread, nrhs, t1, t2)
        loc += 1


print('')
print(df)
print('')

df.to_csv('results_scale_circuit_4.csv')


print('done!')
print('')
