# Input: instance list from the wang-style file, list with  chosen latencies, and the vector of routes that each packet should travel along
import random
import numpy as np
import sys
import noise

def getTimefromPacket(packet):return float(packet.split('\t')[0])
def getDirfromPacket(packet):return int(float(packet.split('\t')[1]))
def getSizefromPacket(packet): #In case packet size is also present as third column in file
	if len(packet.split('\t'))==4:
		return int(float(packet.split('\t')[3]))
	if len(packet.split('\t'))==2:
		return int(float(packet.split('\t')[1]))
	if len(packet.split('\t'))==3:
		return int(float(packet.split('\t')[2]))
	else: return 0


def getWeights(n,alphas):
	aph = alphas.split(',')
	if (len(aph) != n):
		return -1
        vec_alphas = np.array(aph,dtype= np.float)
	w = np.random.dirichlet(vec_alphas,size=1)[0]
	return w



def buildPacket(size,time,direcction):
	return str(time) + '\t' + str(direcction) + '\t' + str(size)

def joingClientServerRoutes(c,s): #joing the routes choosed by client and server into one route to be used by the simulate funtions
	if len(c)!= len(s):
		sys.exit("ERROR: Client and Server routes must have the same length")
	out = []
	for i in xrange(0, len(c)):
		if (c[i]==-1):
			out.append(s[i])
		if (s[i]==-1):
			out.append(c[i])
	return out
	

def simulate(instance,mplatencies,routes):	
	delta = 0.0001 # Delta time to introduce as the time between two cells are sent from the end side
	last_packet = 1 
	last_time = 0	
	delay = 0
	time_last_incomming = 0
	new_trace = []
	for i  in xrange(0,len(instance)): #Iterate over each packet
		last_incomming_cell = 0
		packet = instance[i]
		packet = packet.replace(' ','\t') #For compatibility when data is space sperated not tab separated
		#next_packet = instance[i+1]
		#next_direction = getDirfromPacket(next_packet)
		original_time = getTimefromPacket(packet)
		direction = getDirfromPacket(packet)
		size = getSizefromPacket(packet)
		# Get the route according to the scheme
		route = routes[i]
		# Get the latency for this route
		chosen_latency = float(random.choice(mplatencies[(route%len(mplatencies))]))
		if (last_packet != direction): delay = float(original_time - last_time)/2 #Calculate the RTT/2 (latency) request/response, from the time the out cell is sent till the correspongin incell arrives
		#######################################################################################################################################################
		# original_time - delay = time when the in-cell (measured at client) is on exit
		# original_time - delay + chosen_latency = time when the in-cell is at client after travelling across one of the m circuits.
		# time_last_incomming = time of the las in-cell before the out-cell was on client, it is used + delta to set the time of the outgoing cell
		########################################################################################################################################################
		new_packet = np.array([])			
		if (direction == -1):
			new_packet=[original_time - delay + chosen_latency,direction,size,route]
			#print direction,original_time,original_time - delay + chosen_latency
		if (direction == 1 and last_packet == -1): # If is the first out in the burst, it referes to the last icomming time
			#print direction,original_time,time_last_incomming + delta
			new_packet=[time_last_incomming + delta,direction,size,route]
		if (direction == 1 and last_packet == 1): # If we are in an out burst, refers to the last out
			#print direction,original_time,last_time + delta
			new_packet=[last_time + delta,direction,size,route]
		new_trace.append(new_packet)
		time_last_incomming = original_time - delay + chosen_latency
		last_time = original_time
		last_packet = direction		
	np_new_trace =  np.array(new_trace)
	sorted_new_trace = np_new_trace[np_new_trace[:,0].argsort()] #Sorted according to the new timestamps, this is the final instance applied the multi-path effects
	## Rescale since for a local adversary views the first packet starting at 0
	t_0 = sorted_new_trace[0][0]
	for s in sorted_new_trace:
		s[0] = s[0] - t_0 
	return sorted_new_trace
