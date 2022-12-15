import matplotlib.pyplot as plt
import csv
import numpy as np

task1=True
task2=False
task3=False

files = ['cheetahData/pearl_progress1.csv','cheetahData/pearl_progress2.csv']
files2 = ['cheetahData/progress1_cut.csv','cheetahData/progress2_cut.csv','cheetahData/progress3_cut.csv']

if task1:
    avg_discounted_returns = list()
    for i,file in enumerate(files):
        avg_discounted_returns.append( list() )
        with open(file,'r') as fd:
            reader = csv.reader(fd)
            next(reader)
            trials = 0
            for row in reader:
                trials+=1
                avg_discounted_returns[i].append(float(row[31]))

        # y = np.array(avg_discounted_returns)
        # y_success = np.array(success_rate)
        # x = [i for i in range(1,64)]
    x = [i for i in range(1,trials+1)]
    y = np.mean(avg_discounted_returns,axis=0)
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Discounted Reward')
    plt.title('Average Discounted Reward vs. #Episodes on MetaWorld Picking Task')
    plt.plot(x,y,**{'color':'red','marker':'o' })


    avg_discounted_returns = list()
    for i,file in enumerate(files2):
        avg_discounted_returns.append( list() )
        with open(file,'r') as fd:
            reader = csv.reader(fd)
            next(reader)
            trials = 0
            for row in reader:
                trials+=1
                avg_discounted_returns[i].append(float(row[7]))

        # y = np.array(avg_discounted_returns)
        # y_success = np.array(success_rate)
        # x = [i for i in range(1,64)]
    x2 = [i for i in range(1,trials+1)]
    y2 = np.mean(avg_discounted_returns,axis=0)
    # plt.xlabel('Training Episodes')
    # plt.ylabel('Average Discounted Reward')
    # plt.title('Average Discounted Reward vs. #Episodes on MetaWorld Picking Task')
    # plt.figure()
    plt.plot(x2,y2,**{'color':'blue','marker':'o' })
    plt.legend(['PEARL','Meta-Q-Learning'])

    plt.show()

    # plt.plot(x,y_success,**{'color':'blue','marker':'*' })
    # plt.show()

    # plt.xlabel('# Threads')
    # plt.xticks(x)
    # plt.ylabel('Time(ms)')
    # plt.title('Cluster Execution Time vs. # Threads for N=5040000')
    # plt.legend(['False Sharing','w/out False Sharing'])

if task2:
    y = list()
    y_simd = list()
    with open('task2.out','r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            y.append(float(row[0]))
    with open('task2_simd.out','r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            y_simd.append(float(row[0]))
    y_simd = np.array(y_simd)
    y = np.array(y)
    x = [i for i in range(1,11)]

if task3:
    y = list()
    with open('task3.out','r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            y.append(float(row[0]))
    # y = np.log2(np.array(y))
    # x = [i for i in range(1,26)]
    y = np.array(y)
    x = np.array([2**i for i in range(1,26)])
    result = (x*(8)*1e-9)/(y*1e-3)
    print(result)
