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


engine = './klu_threaded.o'

timeout = 60

###################### specs inclusive ################
max_thread_p = 6  # 2^p
max_nrhs_p = 6  # 2^p
#######################################################


def runSingle(engine, threads, nrhs, reps):
    try:
        filename = 'scircuit/scircuit.mtx'
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
            time1 = float(outLines[-2].split(':')[1])

        else:
            print(p.stderr.decode('utf-8'))
            time1 = -1

    except subprocess.TimeoutExpired:
        time1 = -2

    return time1


#t1 = runSingle(engine, 1, 1)

loc = 0

for pt in range(0, max_thread_p+1):
    for pr in range(0, max_nrhs_p+1):
        thread = 2**pt
        nrhs = 2**pr
        if thread*nrhs <= 64:
            reps = 10
        else:
            reps = 1

        t1 = runSingle(engine, thread, nrhs, reps)
        df.loc[loc, 'index'] = loc
        df.loc[loc, 'threads'] = thread
        df.loc[loc, 'nrhs'] = nrhs
        df.loc[loc, 'time1'] = t1
        print(thread, nrhs, t1)
        loc += 1


print('')
print(df)
print('')

df.to_csv('results.csv')


print('done!')
print('')
