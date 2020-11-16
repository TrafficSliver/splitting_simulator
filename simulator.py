# Code for simulating traffic-splitting strategies
# Used in: https://dl.acm.org/doi/10.1145/3372297.3423351
# Wladimr De la Cadena

#wdlc: Simulator of multipath effect over wang-style instances, IMPORTANT!!! use files in WANG Format with .cell extension. Three colums can be considered for each packet in the .cell file timestamp|direction|size
#Working methods Random, RoundRobin, Weighted Random and Batched Weighted Random and their variation to variable number of paths during the same page load. 

import numpy as np, numpy.random
import sys
import argparse
import glob
import random
from natsort import natsorted
import multipath
import time
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scheme", type=str, help="Splitting scheme", default='round_robin') 
parser.add_argument("-c", "--cells", type=int, help="Cells per circuit in RR (in others it is the initial value)", default=1) 
parser.add_argument("-m", "--circuits", type=int, help="In how many paths is the traffic divided", default=3) 
parser.add_argument("-min", "--circuitsmin", type=int, help="In how many paths is the traffic divided", default=2) 
parser.add_argument("-i", "--inputs", type=str, help="Circuit latencies file", default='circuits_latencies_new.txt') 
parser.add_argument("-o", "--outfolder", type=str, help="Folder for output files", default='outdata') 
parser.add_argument("-w", "--weights", type=str, help="Weights for circuit (comma separated)", default='0.1,0.3,0.6') 
parser.add_argument("-r", "--ranges", type=str, help="Range of cells after of which the wr or wrwc schduler design again", default='10,60') 
parser.add_argument("-p",'--path', nargs='+', help=' fiPath of folder with instancesles (wang_format)')
parser.add_argument("-a", "--alpha", type=str, help="alpha values for the Dirichlet function default np.ones(m)", default='1,1,1')



schemes = ['round_robin','random','weighted_random', 'in_and_out', 'batched_weighted_random', 'bwr_var_paths', 'bwr_var_paths_strict', 'bwr_blocked', 'wr_var_paths', 'rr_var_paths', 'random_var_paths']
bwoh = [0]
def genRRlist(m,length,n):
    out = []
    for i in xrange (0,length):
        for j in xrange(0,n):
            for k in xrange(0,m):
                out.append(j)
    return out

def saveInFile(input_name, inst,r,outfolder):
    numberOfFiles = max(r)+1 # How many files, one per route
    outfiles = []
    for k in xrange(0,numberOfFiles):
        input_name2 = input_name.split('.cell')[0].split('/')[-1]
        out_file_name = outfolder + "/" + input_name2 + "_split_" + str(k) + '.cell'
        outfiles.append(open(out_file_name,'w'))

    jointfilename = outfolder + "/" + input_name.split('.cell')[0].split('/')[-1] + "_join"+ '.cell'
    #jointfile = open (jointfilename,'w')
    for i in xrange(0,len(inst)):
        x_arrstr = np.char.mod('%.15f', inst[i])
        x_arrstr[1] = int(float(x_arrstr[1]))
        outfiles[r[i]].write('\t'.join(x_arrstr) + '\n')
        #jointfile.write('\t'.join(x_arrstr) + '\n')

def saveInFile2(input_name,split_inst,r,outfolder):

    numberOfFiles = max(r)+1 # How many files, one per route
    outfiles = []
    for k in xrange(0,numberOfFiles):
        input_name2 = input_name.split('.cell')[0].split('/')[-1]
        out_file_name = outfolder + "/" + input_name2 + "_split_" + str(k) + '.cell'
        outfiles.append(open(out_file_name,'w'))

    jointfilename = outfolder + "/" + input_name.split('.cell')[0].split('/')[-1] + "_join"+ '.cell'
    jointfile = open (jointfilename,'w')
    for i in xrange(0,len(split_inst)):
        x_arrstr = np.char.mod('%.15f', split_inst[i][:-1])
        x_arrstr[1] = int(float(x_arrstr[1]))
        jointfile.write('\t'.join(x_arrstr) + '\n')

    fs = [0] * numberOfFiles
    ts_o = [0] * numberOfFiles
    for i in xrange(0,len(split_inst)):
        rout = int(split_inst[i][3])
        if (fs[rout] == 0):
            ts_o[rout] = float(split_inst[i][0])
        fs[rout] = 1
        x_arrstr = np.char.mod('%.15f', split_inst[i])
        x_arrstr[1] = int(float(x_arrstr[1]))
        x_arrstr = x_arrstr.astype(float)
        strwrt =  str(x_arrstr[0] - ts_o[rout]) + '\t' + str(int(x_arrstr[1])) + '\t' + str(x_arrstr[2])
        outfiles[rout].write(strwrt+ '\n')		

def getCircuitLatencies(l,n):
    file_latencies = open(l,'r')
    row_latencies = file_latencies.read().split('\n')[:-1]
    numberOfClients = int(row_latencies[-1].split(' ')[0])
    randomclient = random.randint(1,numberOfClients)
    ## Get the multiple circuits of the selected client:
    multipath_latencies = []
    for laten in row_latencies:
        clientid = int(laten.split(' ')[0])
        if (clientid == randomclient):
            multipath_latencies.append(laten.split(' ')[2].split(','))	
    ## I only need n circuits, it works when n <  number of circuits in latency file (I had max 6)
    multipath_latencies = multipath_latencies[0:n]
    return multipath_latencies

def sim_random(n,latencies,traces,outfiles):
    print "Simulating Random multi-path scheme..."
    traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    for instance_file in traces_file: 
        instance = open(instance_file,'r')
        instance = instance.read().split('\n')[:-1]
        # Get n circuits latencies as a multipath virtual structure of the same client for this instance
        mplatencies = getCircuitLatencies(latencies, n)	# it is a list of list of latencies for each of m circuits. length = m
        routes = [random.randrange(0, n, 1) for _ in range(len(instance))] # Random routes
        new_instance = multipath.simulate(instance,mplatencies,routes) # Simulate the multipath effect for the given latencies and routes
        saveInFile(instance_file,new_instance,routes,outfiles) # Save the transformed current instance according to thei respective routes

def sim_round_robin(n,latencies,traces,outfiles,cpercirc):
    print "Simulating Round Robin multi-path scheme..."
    traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    for instance_file in traces_file: 
        instance = open(instance_file,'r')
        instance = instance.read().split('\n')[:-1]
        # Get n circuits latencies as a multipath virtual structure of the same client for this instance
        mplatencies = getCircuitLatencies(latencies, n)	# it is a list of list of latencies for each of m circuits. length = m
        routes = genRRlist(cpercirc,len(instance),n) # Generate Round Robin routes 
        new_instance = multipath.simulate(instance,mplatencies,routes) # Simulate the multipath effect for the given latencies and routes
        saveInFile2(instance_file,new_instance,routes,outfiles) # Save the transformed current instance according to thei respective routes


def sim_weighted_random(n,latencies,traces,outfiles,weights,alphas):
    print "Simulating Weighted Random multi-path scheme... alphas:", alphas
    traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    for instance_file in traces_file: 
        instance = open(instance_file,'r')
        instance = instance.read().split('\n')[:-1]
        w_out = multipath.getWeights(n,alphas)
        w_in = multipath.getWeights(n,alphas)
        # Get n circuits latencies as a multipath virtual structure of the same client for this instance
        mplatencies = getCircuitLatencies(latencies, n)	# it is a list of list of latencies for each of m circuits. length = m
        # for each instance I need to create a new randomly created weights vector
        routes_server = []
        routes_client = []
        last_client_route =  np.random.choice(np.arange(0,n),p = w_out)
        last_server_route = np.random.choice(np.arange(0,n),p = w_in)
        routes = []
        for i in xrange(0,len(instance)):
            packet = instance[i]
            packet = packet.replace(' ','\t') #For compatibility when data is space sperated not tab separated
            direction = multipath.getDirfromPacket(packet)			
            if (direction == 1):
                    routes_server.append(-1) # Just to know that for this packet the exit does not decide the r$
                    last_client_route =  np.random.choice(np.arange(0,n),p = w_out)
                    routes_client.append(last_client_route)

            if (direction == -1):
                    routes_client.append(-1) # Just to know that for this packet the client does not decide the$
                    last_server_route =  np.random.choice(np.arange(0,n),p = w_in)
                    routes_server.append(last_server_route)

        routes = multipath.joingClientServerRoutes(routes_client,routes_server)
        ##### Routes Created, next to the multipath simulation
        new_instance = multipath.simulate(instance,mplatencies,routes) # Simulate the multipath effect for the give$
        saveInFile2(instance_file,new_instance,routes,outfiles)


def sim_in_and_out(n,latencies,traces,outfiles):
    print "Simulating In and Out multi-path scheme..."
    traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    # By default ind and out makes only one out traffic and n-1 in traffic in a random manner
    for instance_file in traces_file: 
        instance = open(instance_file,'r')
        instance = instance.read().split('\n')[:-1]
        # Get n circuits latencies as a multipath virtual structure of the same client for this instance
        mplatencies = getCircuitLatencies(latencies, n)	# it is a list of list of latencies for each of m circuits. length = m
        routes = []
        routes_client = []
        routes_server = []
        for i in xrange(0,len(instance)):
            packet = instance[i]
            packet = packet.replace(' ','\t') #For compatibility when data is space sperated not tab separated
            direction = multipath.getDirfromPacket(packet)

            if (direction == 1): # if it is outgoing set a fixed route
                routes_server.append(-1) # Just to know that for this packet the exit does not decide the route
                routes_client.append(0) 
            if (direction == -1): # if it is incomming just sent through a random route, it'd better trying to have same amount of 
                routes_server.append(1) # Just to know that for this packet the exit does not decide the route
                routes_client.append(-1)
        routes = multipath.joingClientServerRoutes(routes_client,routes_server)
        ##### Routes Created, next to the multipath simulation
        new_instance = multipath.simulate(instance,mplatencies,routes) # Simulate the multipath effect for the given latencies and routes
        saveInFile2(instance_file,new_instance,routes,outfiles)

def sim_bwr(n,latencies,traces,outfiles,range_, alphas):
    print "Simulating BWR multi-path scheme..."
    traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    ranlow = int(range_.split(',')[0])
    ranhigh = int(range_.split(',')[1])

    for instance_file in traces_file:
        w_out = multipath.getWeights(n, alphas)
        w_in = multipath.getWeights(n, alphas)
        instance = open(instance_file,'r')
        instance = instance.read().split('\n')[:-1]
        mplatencies = getCircuitLatencies(latencies, n)	# it is a list of list of latencies for each of m circuits. length = m
        routes_client = []
        routes_server = []
        sent_incomming = 0
        sent_outgoing = 0
        last_client_route =  np.random.choice(np.arange(0,n),p = w_out)
        last_server_route = np.random.choice(np.arange(0,n),p = w_in)
        for i in xrange(0,len(instance)):
            packet = instance[i]
            packet = packet.replace(' ','\t') #For compatibility when data is space sperated not tab separated
            direction = multipath.getDirfromPacket(packet)

            if (direction == 1):
                routes_server.append(-1) # Just to know that for this packet the exit does not decide the route
                sent_outgoing += 1
                C = random.randint(ranlow,ranhigh) #After how many cells the scheduler sets new weights
                routes_client.append(last_client_route) 
                if (sent_outgoing % C == 0): #After C cells are sent, change the circuits
                            last_client_route =  np.random.choice(np.arange(0,n),p = w_out)

            if (direction == -1): 
                routes_client.append(-1) # Just to know that for this packet the client does not decide the route
                routes_server.append(last_server_route)
                sent_incomming += 1
                C = random.randint(ranlow,ranhigh) #After how many cells the scheduler sets new weights
                if (sent_incomming % C == 0): #After C cells are sent, change the circuits
                     last_server_route = np.random.choice(np.arange(0,n),p = w_in)


        routes = multipath.joingClientServerRoutes(routes_client,routes_server)
        ##### Routes Created, next to the multipath simulation
        new_instance = multipath.simulate(instance,mplatencies,routes) # Simulate the multipath effect for the given latencies and routes
        saveInFile2(instance_file,new_instance,routes,outfiles)
        
def sim_bwr_blocked(n,latencies,traces,outfiles,range_):
    print "Simulating BWR multi-path scheme blocking last selected route..."
    print traces
    traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    ranlow = int(range_.split(',')[0])
    ranhigh = int(range_.split(',')[1])

    for instance_file in traces_file:
        w_out = multipath.getWeights(n)
        w_in = multipath.getWeights(n)
        instance = open(instance_file,'r')
        instance = instance.read().split('\n')[:-1]
        mplatencies = getCircuitLatencies(latencies, n)	# it is a list of list of latencies for each of m circuits. length = m
        routes_client = []
        routes_server = []
        sent_incomming = 0
        sent_outgoing = 0
        last_client_route =  np.random.choice(np.arange(0,n),p = w_out)
        last_server_route = np.random.choice(np.arange(0,n),p = w_in)

        for i in xrange(0,len(instance)):
            packet = instance[i]
            packet = packet.replace(' ','\t') #For compatibility when data is space sperated not tab separated
            direction = multipath.getDirfromPacket(packet)

            if (direction == 1):
                routes_server.append(-1) # Just to know that for this packet the exit does not decide the route
                sent_outgoing += 1
                C = random.randint(ranlow,ranhigh) #After how many cells the scheduler sets new weights
                routes_client.append(last_client_route) 
                if (sent_outgoing % C == 0): #After C cells are sent, change the circuits
                    while(1):
                        client_route =  np.random.choice(np.arange(0,n),p = w_out)
                        if (client_route != last_client_route):
                            last_client_route = client_route
                            break # In this way, we block the possibility of choosing the same circuit previously chosen
                            
            if (direction == -1): 
                routes_client.append(-1) # Just to know that for this packet the client does not decide the route
                routes_server.append(last_server_route)
                sent_incomming += 1
                C = random.randint(ranlow,ranhigh) #After how many cells the scheduler sets new weights
                if (sent_incomming % C == 0): #After C cells are sent, change the circuits
                    while(1):
                        server_route =  np.random.choice(np.arange(0,n),p = w_in)
                        if (server_route != last_server_route):
                            last_server_route = server_route
                            break # In this way, we block the possibility of choosing the same circuit previously chosen

        routes = multipath.joingClientServerRoutes(routes_client,routes_server)
        ##### Routes Created, next to the multipath simulation
        new_instance = multipath.simulate(instance,mplatencies,routes) # Simulate the multipath effect for the given latencies and routes
        saveInFile2(instance_file,new_instance,routes,outfiles)


def sim_bwr_var_paths(n,nmin,latencies,traces,outfiles,range_, alphas):
    print "Simulating bwr with variable number of paths", nmin, n
    traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    ranlow = int(range_.split(',')[0])
    ranhigh = int(range_.split(',')[1])
    for instance_file in traces_file:
	## First lets choose the number of circuits randomly
        n_random = random.randint(nmin,n)
        alphas = np.ones(n_random)
        alphas = alphas.astype(str)
        alphas = ','.join(alphas)
        w_out = multipath.getWeights(n_random, alphas)
        w_in = multipath.getWeights(n_random, alphas)
        instance = open(instance_file,'r')
        instance = instance.read().split('\n')[:-1]
        mplatencies = getCircuitLatencies(latencies, n_random)
        routes_client = []
        routes_server = []
        sent_incomming = 0
        sent_outgoing = 0
        last_client_route =  np.random.choice(np.arange(0,n_random),p = w_out)
        last_server_route = np.random.choice(np.arange(0,n_random),p = w_in)
        #C = random.randint(ranlow,ranhigh) #After how many cells the scheduler sets new weights
        for i in xrange(0,len(instance)):
            packet = instance[i]
            packet = packet.replace(' ','\t') #For compatibility when data is space sperated not tab separated
            direction = multipath.getDirfromPacket(packet)
            if (direction == 1):
                routes_server.append(-1) # Just to know that for this packet the exit does not decide the r$
                sent_outgoing += 1
                C = random.randint(ranlow,ranhigh) #After how many cells the scheduler sets new weights
                routes_client.append(last_client_route)
                if (sent_outgoing % C == 0): #After C cells are sent, change the circuits
                    last_client_route =  np.random.choice(np.arange(0,n_random),p = w_out)
            if (direction == -1):
                routes_client.append(-1) # Just to know that for this packet the client does not decide the$
                routes_server.append(last_server_route)
                sent_incomming += 1
                C = random.randint(ranlow,ranhigh) #After how many cells the scheduler sets new weights
                if (sent_incomming % C == 0): #After C cells are sent, change the circuits
                    last_server_route = np.random.choice(np.arange(0,n_random),p = w_in) 

        routes = multipath.joingClientServerRoutes(routes_client,routes_server)
        #### Routes Created, next to the multipath simulation
        new_instance = multipath.simulate(instance,mplatencies,routes) # Simulate the multipath effect for the give$
        saveInFile2(instance_file,new_instance,routes,outfiles)


def sim_bwr_var_paths_strict(n,nmin,latencies,traces,outfiles,range_):
    ## Take the dataset and do: m=2 to 20%, m=3 to 20%, m=4 to 20% and m=5 to 20% of the whole dataset.
    ## It means that e.g., BWR2-5 Strict has almost 20% of instances split into two parts, 20% into three and so on
    ## BWR 4-5 Strict schould have 50/50 % to m=4 and m=5. However, experimentally we obtained couple of dozens of samples divided into m=2, and m=3.
    print "Simulating bwr with variable number of paths strict to avoid less than nmin Cmax and Cmind should be reduced to 15-20"
    traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    ranlow = int(range_.split(',')[0])
    ranhigh = int(range_.split(',')[1])
    number_of_traces = len(traces_file)
    #print number_of_traces, n, nmin
    chunks_dataset = 100 / (n - nmin + 1)
    n_sel = []
    #print chunks_dataset
    for i in range(nmin,n+1):
            n_sel.extend([i] * chunks_dataset)
    #print n_sel, len(n_sel)
    #print n_sel
    kk = 0
    for instance_file in traces_file:
        ## First lets choose the number of circuits randomly
        #print instance_file, "divided into",
        n_random =  n_sel[kk%len(n_sel)]
        kk += 1 
        #print n_random, instance_file
        w_out = multipath.getWeights(n_random)
        w_in = multipath.getWeights(n_random)
        instance = open(instance_file,'r')
        instance = instance.read().split('\n')[:-1]
        mplatencies = getCircuitLatencies(latencies, n_random)
        routes_client = []
        routes_server = []
        sent_incomming = 0
        sent_outgoing = 0
        last_client_route =  np.random.choice(np.arange(0,n_random),p = w_out)
        last_server_route = np.random.choice(np.arange(0,n_random),p = w_in)
        #C = random.randint(ranlow,ranhigh) #After how many cells the scheduler sets new weights
        for i in xrange(0,len(instance)):
            packet = instance[i]
            packet = packet.replace(' ','\t') #For compatibility when data is space sperated not tab separated
            direction = multipath.getDirfromPacket(packet)
            if (direction == 1):
                routes_server.append(-1) # Just to know that for this packet the exit does not decide the r$
                sent_outgoing += 1
                C = random.randint(ranlow,ranhigh) #After how many cells the scheduler sets new weights
                routes_client.append(last_client_route)
                if (sent_outgoing % C == 0): #After C cells are sent, change the circuits
                    while(1):
                        client_route =  np.random.choice(np.arange(0,n_random),p = w_out)
                        if (client_route != last_client_route):
                            last_client_route = client_route
                            break # In this way, we block the possibility of choosing the same circuit previously chosen
                    
            if (direction == -1):
                routes_client.append(-1) # Just to know that for this packet the client does not decide the$
                routes_server.append(last_server_route)
                sent_incomming += 1
                C = random.randint(ranlow,ranhigh) #After how many cells the scheduler sets new weights
                if (sent_incomming % C == 0): #After C cells are sent, change the circuits
                    while(1):
                        server_route =  np.random.choice(np.arange(0,n_random),p = w_in)
                        if (server_route != last_server_route):
                            last_server_route = server_route
                            break # In this way, we block the possibility of choosing the same circuit previously chosen
        
        routes = multipath.joingClientServerRoutes(routes_client,routes_server)
        #### Routes Created, next to the multipath simulation
        new_instance = multipath.simulate(instance,mplatencies,routes) # Simulate the multipath effect for the give$
        saveInFile2(instance_file,new_instance,routes,outfiles)

        

def sim_wr_var_paths(n,nmin,latencies,traces,outfiles):
    print "Simulating weighted-random with variable number of paths", nmin, n        
    traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    for instance_file in traces_file: 
        ## First lets choose the number of circuits randomly
        n_random = random.randint(nmin,n)
        #print n_random, instance_file
        w_out = multipath.getWeights(n_random)
        w_in = multipath.getWeights(n_random)
        instance = open(instance_file,'r')
        instance = instance.read().split('\n')[:-1]
        # Get n circuits latencies as a multipath virtual structure of the same client for this instance
        mplatencies = getCircuitLatencies(latencies, n_random)	# it is a list of list of latencies for each of m circuits. length = m
        # for each instance I need to create a new randomly created weights vector
        routes_server = []
        routes_client = []
        last_client_route =  np.random.choice(np.arange(0,n_random),p = w_out)
        last_server_route = np.random.choice(np.arange(0,n_random),p = w_in)
        routes = []
        for i in xrange(0,len(instance)):
            packet = instance[i]
            packet = packet.replace(' ','\t') #For compatibility when data is space sperated not tab separated
            direction = multipath.getDirfromPacket(packet)
            if (direction == 1):
                    routes_server.append(-1) # Just to know that for this packet the exit does not decide the r$
                    last_client_route =  np.random.choice(np.arange(0,n_random),p = w_out)
                    routes_client.append(last_client_route)

            if (direction == -1):
                    routes_client.append(-1) # Just to know that for this packet the client does not decide the$
                    last_server_route =  np.random.choice(np.arange(0,n_random),p = w_in)
                    routes_server.append(last_server_route)

        routes = multipath.joingClientServerRoutes(routes_client,routes_server)
        ##### Routes Created, next to the multipath simulation
        new_instance = multipath.simulate(instance,mplatencies,routes) # Simulate the multipath effect for the give$
        saveInFile2(instance_file,new_instance,routes,outfiles)
      

def sim_rr_var_paths(n,nmin,latencies,traces,outfiles,cpercirc):
    print "Simulating round_robin with variable number of paths", nmin, n        
    traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    for instance_file in traces_file:
        n_random = random.randint(nmin,n)
        instance = open(instance_file,'r')
        instance = instance.read().split('\n')[:-1]
        # Get n circuits latencies as a multipath virtual structure of the same client for this instance
        mplatencies = getCircuitLatencies(latencies, n_random)	# it is a list of list of latencies for each of m circuits. length = m
        routes = genRRlist(cpercirc,len(instance),n_random) # Generate Round Robin routes 
        new_instance = multipath.simulate(instance,mplatencies,routes) # Simulate the multipath effect for the given latencies and routes
        saveInFile2(instance_file,new_instance,routes,outfiles) # Save the transformed current instance according to their respective routes
        
def sim_random_var_paths(n,nmin,latencies,traces,outfiles):
    print "Simulating Random with variable paths multi-path scheme..."
    traces_file = natsorted(glob.glob(traces[0]+'/*.cell'))
    for instance_file in traces_file:
        n_random = random.randint(nmin,n)
        instance = open(instance_file,'r')
        instance = instance.read().split('\n')[:-1]
        # Get n circuits latencies as a multipath virtual structure of the same client for this instance
        mplatencies = getCircuitLatencies(latencies, n_random)	# it is a list of list of latencies for each of m circuits. length = m
        routes = [random.randrange(0, n_random, 1) for _ in range(len(instance))] # Random routes
        new_instance = multipath.simulate(instance,mplatencies,routes) # Simulate the multipath effect for the given latencies and routes
        saveInFile(instance_file,new_instance,routes,outfiles) # Save the transformed current instance according to thei respective routes
        
if __name__ == '__main__':
    args = parser.parse_args()
    scheme_ = args.scheme
    if (scheme_ not in schemes):
        sys.exit("ERROR: Splitting scheme not supported")
    cells_per_circuit_ = args.cells
    paths_ = args.circuits
    paths_min = args.circuitsmin
    latencies_ = args.inputs
    traces_ = args.path
    outfolder_ = args.outfolder
    weights_ = args.weights
    range_ = args.ranges
    alpha_ = args.alpha
    val = 0
    starttime = time.time()
    if (scheme_ == 'random'):
        sim_random(paths_, latencies_,traces_,outfolder_)

    if (scheme_ == 'round_robin'):
        sim_round_robin(paths_, latencies_,traces_,outfolder_, cells_per_circuit_)

    if (scheme_ == 'weighted_random'):
        sim_weighted_random(paths_, latencies_,traces_,outfolder_, weights_, alpha_)

     if (scheme_ == 'in_and_out'):
        sim_in_and_out(paths_, latencies_,traces_,outfolder_)

    if (scheme_ == 'batched_weighted_random'):
        sim_bwr(paths_, latencies_,traces_,outfolder_,range_, alpha_)

    if (scheme_ == 'bwr_var_paths'): 
        sim_bwr_var_paths(paths_, paths_min, latencies_,traces_,outfolder_,range_, alpha_)

    if (scheme_ == 'bwr_var_paths_strict'): 
        sim_bwr_var_paths_strict(paths_, paths_min, latencies_,traces_,outfolder_,range_)
       
    if (scheme_ == 'bwr_blocked'):
        sim_bwr_blocked(paths_, latencies_,traces_,outfolder_,range_)
        
    if (scheme_ == 'wr_var_paths'): 
        sim_wr_var_paths(paths_, paths_min, latencies_,traces_,outfolder_)
        
    if (scheme_ == 'rr_var_paths'): 
        sim_rr_var_paths(paths_, paths_min, latencies_,traces_,outfolder_,cells_per_circuit_)
            
    if (scheme_ == 'random_var_paths'): 
        sim_random_var_paths(paths_, paths_min, latencies_,traces_,outfolder_)
        
        
    endtime = time.time()
    print "Multi-path Simulation done!!! I took (s):", (endtime - starttime)
    print "bandwidth OH: ", val
