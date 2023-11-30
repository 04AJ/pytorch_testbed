import os
import sys
import optparse
import random

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
	from sumolib import checkBinary
	import traci
	import sumolib
else:
	sys.exit("No environment variable SUMO_HOME!")

class sumo_env:

	def __init__(self):
		'''
			args:
				out_dict:		type: dict of dict, a 2D dictionary that stores the connection of the network.
				length_dict:	type: dict, stores the length of each edge.
				choice_list:	type: list, a constant list taht shows the available choices
				edge_now:		type: string, a string for the id of edge the vehicle is now on
				target_vehicle:	type: string, a string for the id of the vehicle
				final_target:	type: string, a string for the id of the edge the vehicle aims at
				last_edge:		type: string, a string for the id of the edge the vehicle was on during last step.
				start_edge:		type: string, a string for the id of the edge based on which the vehicle makes a choice.
				index_dict:		type: dict, maps edge id to an index
		'''
		net_file_name  = "test4.net.xml"
		[self.length_dict, self.out_dict, self.index_dict] = self.getConnectionInfo(net_file_name)
		#self.choice_list = ["l", "L", "R", "r", "s", "t"]
		self.choice_list = ["s", "t", "R", "r", "L", "l"]
		self.edge_now = ""
		self.target_vehicle = "0"
		self.final_target = "destination"#modified for the small-scale map already
		self.last_edge = ""
		self.start_edge = ""
		self.state_size = 1+len(self.choice_list)
		self.action_size = len(self.choice_list)


	def reset(self):
		#reset the enviornment here and invokes sumo to start a new round
		self.last_edge = ""
		# sumoBinary = checkBinary('sumo-gui')
		sumoBinary = checkBinary('sumo')
		traci.start([sumoBinary, "-c", "myconfig.sumocfg",
				"--tripinfo-output", "trips.trips.xml", "--message-log", "sim_message.log", "--duration-log.disable", "true"])
		arrived, steps = self.simulate_step()
		traci.vehicle.setMaxSpeed(self.target_vehicle, 11)
		if arrived:
			print("Target reached when initialization")
		self.start_edge = self.edge_now
		state = self.getState(self.start_edge)
		return state

	def getState(self, en):
		state = []
		state.append(self.index_dict[en])
		for c in self.choice_list:
			if c in self.out_dict[en].keys():
				state.append(1)
			else:
				state.append(0)
		return state

		# action is just an index (random on intelligent) to select an item from choice_list
	def step(self, action):
		#return value is: state, reward, done_flag, 
		#print("Action received: "+str(action))
		choice_now = self.choice_list[action]

		# if vehicle makes an invalid choice (chooses invalid edge)--> reward is -1000
		if choice_now not in self.out_dict[self.start_edge].keys():
			#input("*******Invalid choice**************")

			state = self.getState(self.start_edge)
			reward = -10000
			done_flag = True
			info = self.edge_now
			traci.close()
			return state, reward, done_flag, info
		
		# selecting edge for vehicle to target
		target_edge = self.out_dict[self.start_edge][choice_now]
		traci.vehicle.changeTarget(self.target_vehicle, target_edge)
		#print("Set target as "+target_edge)
		#when to confirm the final_target?

		# target doesn't reach destiantion so reward = -steps
		if self.length_dict[target_edge]<=10 and target_edge!=self.final_target:
			#input("*********short_destination************")
			self.start_edge = target_edge
			state = self.getState(self.start_edge)
			reward = -1
			done_flag = False
			info = "Too short target edge confronted."
			return state, reward, done_flag, info
		#input("*********simulate************")
		arrived, steps = self.simulate_step()
		done_flag = arrived
		self.start_edge = self.edge_now
		state = self.getState(self.start_edge)
		reward = -steps
		if arrived is True:
			reward = 10000
		info = ""
		return state, reward, done_flag, info

	def simulate_step(self):
		step = 0
		while traci.simulation.getMinExpectedNumber()>0:
			traci.simulationStep()
			step += 1
			id_list = set(traci.vehicle.getIDList())
			if self.target_vehicle in id_list:
				temp_edge_now = traci.vehicle.getRoadID(self.target_vehicle)
				#print(temp_edge_now)
				if temp_edge_now == self.final_target:
					self.edge_now = temp_edge_now
					traci.close()
					return True, step

				if temp_edge_now in self.out_dict.keys() and temp_edge_now!=self.last_edge:
					self.edge_now = temp_edge_now
					self.last_edge = self.edge_now

					return False, step
			else:
				arrived_list = set(traci.simulation.getArrivedIDList())
				if self.target_vehicle in arrived_list:
					#self.edge_now = traci.vehicle.getRoadID(self.target_vehicle)
					#print(self.edge_now)
					print("Target arrives at some points.")
					return True, step


# instiantials out_dict, length_dict and index_dict
	def getConnectionInfo(self, net_file_name):
		net = sumolib.net.readNet(net_file_name)
		out_dict = {}
		length_dict = {}
		index_dict = {}
		counter = 0
		all_edges = net.getEdges()
		for edge_now in all_edges:
			edge_now_id = edge_now.getID()
			if edge_now_id in index_dict.keys():
				print(edge_now_id+" already exists!")
			else:
				index_dict[edge_now_id] = counter
				counter += 1
			if edge_now_id in out_dict.keys():
				print(edge_now_id+" already exists!")
			else:
				out_dict[edge_now_id] = {}
			if edge_now_id in length_dict.keys():
				print(edge_now_id+" already exists!")
			else:
				length_dict[edge_now_id] = edge_now.getLength()
			#edge_now is sumolib.net.edge.Edge
			out_edges = edge_now.getOutgoing()
			for out_edge_now in out_edges:
				if not out_edge_now.allows("passenger"):
					#print("Found some roads prohibited")
					continue
				conns = edge_now.getConnections(out_edge_now)
				for conn in conns:
					dir_now = conn.getDirection()
					out_dict[edge_now_id][dir_now] = out_edge_now.getID()

		return [length_dict, out_dict, index_dict]


#codes checking connections
#env=sumo_env()
#for key, index in env.index_dict.items():
#	if index == 494:
#		print(key)
#		break
#out_edges = env.out_dict["423505993"]
#for key, out_edge in out_edges.items():
#	print(key+': '+out_edge)

#test code
'''
env = sumo_env()
edge_now = env.reset()
arrived = False
while not arrived:
	#interact with step here. check whether it can randomly walk first then check the minus reward.
	dice = random.randint(0,5)
	initial = dice
	while env.choice_list[dice] not in env.out_dict[edge_now].keys():
		dice += 1
		dice %= 6
		if dice == initial:
			print("Dead End!")
			break
	state, reward, arrived, info = env.step(dice)
	print(state)
	print(reward)
	print(arrived)
	print(info)
	print("********************")
	edge_now = state
'''



