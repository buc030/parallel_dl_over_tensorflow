import tensorflow as tf

import argparse

parser = argparse.ArgumentParser(description='Run a server starting from port 2222.')
parser.add_argument('-i', type=int, nargs=1, required=True, help='server index')
args = parser.parse_args()


#server has a master and a worker.
#The master handles sess.run and stuff
#While the worker run the actual graphs.

#I could also have just one server and connect to it from many clients, each client will run an experiemnt.

#Start num_servers servers
num_servers = 1
servers_addr = []
for i in range(num_servers):
	server_addr = "localhost:" + str(2222 + i)
	servers_addr.append(server_addr)
print servers_addr

cluster = tf.train.ClusterSpec({"local": servers_addr})

servers = []
#for i in range(len(servers_addr)):
    #log_device_placement
    #allow_soft_placement=True
    #sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,
    #                       device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])
    
    #config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
config.allow_soft_placement = True

servers.append(tf.train.Server(cluster, job_name="local", task_index=args.i[0], config=config))
#server0 = 
#server1 = tf.train.Server(cluster, job_name="local", task_index=1)


#print server0.target
#print server0.target
print [s.target for s in servers]


for s in servers:
    s.join()
