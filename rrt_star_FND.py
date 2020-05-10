import numpy as np
from Obstacle import Obstacle
from Tree_3 import Tree
import utils
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


#########################################
############### Task Setup ##############
#########################################
start = [0,0]
goal = [10,10]
xmin, ymin, xmax, ymax = -15,-15,15,15 #grid world borders
obst1 = Obstacle('rect',[3, 3, 3,3], [0,0], np.eye(2))
obst2 = Obstacle('circle',[2,5,2], [0,0], np.eye(2))
obst3 = Obstacle('rect', [8,-4,1,4], [0,0], np.eye(2))
obstacles = [obst1, obst2, obst3] #list of obstacles
N = 2000 #number of iterations
epsilon = 0.5 #near goal tolerance
maxNumNodes = 1000 #upper limit on tree size 
eta = 1.0 #max branch length
gamma = 20.0 #param to set for radius of hyperball
goalFound = False
#########################################
#for plotting
# iterations = []
# costs = []
# path = [];
fig, ax = plt.subplots()
plt.ylim((-15,15))
plt.xlim((-15,15))
ax.set_aspect('equal', adjustable='box')
#########################################


#########################################
########### Begin Iterations ############
#########################################
#1. Initialize Tree and growth
print("Initializing FN TREE.....")
tree = Tree(start, goal, obstacles, xmin,ymin,xmax, ymax)
#2. Set pcurID = 0; by default in Tree instantiation
#3. Get Solution Path
solPath, solPathID = tree.initGrowth()
####################
#4. Init movement()-->> update pcurID 
tree.pcurID = solPathID[1]
####################
#5. Begin replanning loop (and move obstacles?), while pcur is not goal, do...
# while np.linagl.norm(tree.nodes[tree.pcurID, 0:2] - goal) > epsilon:
for __ in range(1):
	#6. Obstacle Updates
	# tree.updateObstacles()
	tree.obstacles[0].position = np.array([ 7, 2 ])
	#7. if solPath breaks
	if tree.detectCollision(solPath):
		#8. Stop Movement
		#9. select remaining valid branches
		solPathID = tree.selectBranch(tree.pcurID, solPathID)
		print("********************************************************")
		print("********* Path Breaks, collision detected! *************")
		print("********************************************************")
		# break
		#10. Separate tree
		separatePathID, orphanedTree = tree.validPath(solPathID)
		separatePath = orphanedTree[separatePathID, 0:2].reshape(-1,2)
		#.11-20 Try to reconnect main with orphanedTree
		reconnected = tree.reconnect(separatePathID)

		if reconnected:
			print('\n')
			print("				RECONNECT SUCCESSFUL !		")
			print('\n')
			utils.drawTree(tree.nodes, ax, 'blue')
		
		else:
			print('\n')
			print("				RECONNECT FAILED!		")
			print('\n')
			#21. if reconnect fails, regrow tree
			solPath, solPathID = tree.regrow()

			print('\n')
			print("				REGROW SUCCESSFUL !		")
			print('\n')
			utils.drawTree(tree.nodes, ax, 'grey')
		
		utils.drawTree(orphanedTree, ax, 'purple')
		utils.drawPath(separatePath, ax, 'green')

#### DRAW ENVIRONMENT ######
for obs in tree.obstacles:
	utils.drawShape(obs.toPatch(), ax)
G = Obstacle('circle', [goal[0], goal[1], 0.3], [0,0], 0*np.eye(2))
S = Obstacle('circle', [start[0], start[1], 0.3], [0,0], 0*np.eye(2))
utils.drawShape(G.toPatch('red'), ax)
utils.drawShape(S.toPatch('pink'), ax)

plt.show()