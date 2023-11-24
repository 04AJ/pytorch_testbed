import os
import sys
import optparse
import random

choice_list = ["l", "L", "R", "r", "s", "t"]
if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("No environment variable SUMO_HOME!")

from sumolib import checkBinary
import traci
import sumolib

def run():
	while traci.simulation.getMinExpectedNumber() > 0:
		traci.simulationStep()
		id_list = set(traci.vehicle.getIDList())
		target_vehicle = "0"
		target_edge = "closed"
		if target_vehicle in id_list:
			traci.vehicle.changeTarget(target_vehicle, target_edge)

	traci.close()

sumoBinary = checkBinary('sumo-gui')
traci.start([sumoBinary, "-c", "myconfig.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
run()