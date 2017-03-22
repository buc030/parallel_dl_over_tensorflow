
import argparse
import pickle
from experiment import *


#This script should be run with only one visible GPU.
parser = argparse.ArgumentParser(description='Run an experiment.')
#parser.add_argument('--gpu_num', type=int, nargs=1, required=True, help='The gpu number these threads will use')
parser.add_argument('--start', type=int, nargs=1, required=True, help='The index of the experiment to start from')
parser.add_argument('--count', type=int, nargs=1, required=True, help='How many experiments to run')
parser.add_argument('--name', nargs=1, required=True, help='The experiemnt name')
args = parser.parse_args()


def get_gpu():
    return 0


def run_experiment_wrapper(bs, sesop_freq, hSize, epochs, file_writer_suffix):
    file_writer_suffix = args.name[0] + '/' + file_writer_suffix
    dev = get_gpu()
    with open('/tmp/generated_data/' + file_writer_suffix + '.descr', 'wb') as f:
        pickle.dump(['started', [bs, sesop_freq, hSize, epochs, dev]], f)


    print 'using device = ' + str(dev)
    with tf.device('/gpu:' + str(dev)):
        run_experiment(bs, sesop_freq, hSize, epochs, file_writer_suffix)

    with open('/tmp/generated_data/' + file_writer_suffix + '.descr', 'wb') as f:
        pickle.dump(['done', [bs, sesop_freq, hSize, epochs, dev]], f)


def get_dones(counter):
    res = []
    for i in range(counter):
        try:
            with open('/tmp/generated_data/' + args.name[0] + '/' + str(i) + '.descr', 'rb') as f:
                r = pickle.load(f)
                if r[0] == 'done':
                #if r[0] == 'started' or r[0] == 'done':
                #    print str(i) + ' started but not done'
                    res.append(r[1][0:-1])
                print str(i) + ':' + str(pickle.load(f))
        except:
            pass
    return res

def run_all_experiments():

    bs=[]
    sesop_freq=[]
    hSize=[]
    epochs=[]
    file_writer_suffix=[]
    counter = 0
    acc_counter = -1
    for _bs in [1, 5, 10, 100]: #4
        for _sesop_freq in [0.01, 0.1, 0.5, 0.9, 0.99]: #5
            for _h in [0, 1, 2, 4, 8, 16]: #6

                acc_counter += 1
                if acc_counter < args.start[0]:
                    continue

                if counter >= args.count[0]:
                    break

                bs.append(_bs)
                hSize.append(_h)
                sesop_freq.append(_sesop_freq)
                epochs.append(10)
                file_writer_suffix.append(str(acc_counter))
                counter += 1

    threads = []
    for i in range(len(bs)):
        if [bs[i], sesop_freq[i], hSize[i], epochs[i]] in get_dones(4*6*5):
            print 'Skipping ' + str([bs[i], sesop_freq[i], hSize[i], epochs[i]]) + ' coz it ran/started'
            #continue
        print 'Running ' + str([bs[i], sesop_freq[i], hSize[i], epochs[i]])
        #continue
        t = threading.Thread(target=run_experiment_wrapper,\
        args = (bs[i], sesop_freq[i], hSize[i], epochs[i], file_writer_suffix[i]))
        t.start()
        threads.append(t)


    for t in threads:
        continue
        t.join()


run_all_experiments()

