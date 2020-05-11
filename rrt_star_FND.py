import numpy as np
from Obstacle import Obstacle
from Tree import Tree
import utils
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import cv2 as cv


#########################################
############### Task Setup ##############
#########################################
start = [-5,-5]
goal = [10,10]
xmin, ymin, xmax, ymax = -15,-15,15,15 #grid world borders
obst1 = Obstacle('rect',[2, 2, 2,3], [-0.8,-0.5], 5*np.eye(2))
obst2 = Obstacle('circle',[0,9,2], [-0.5,0.5], 5*np.eye(2))
obst3 = Obstacle('rect', [8,-4,1,4], [0,0], np.eye(2))
obst4 = Obstacle('rect', [-1,-2,7,1], [0,0], np.eye(2))

obstacles = [obst1, obst2, obst3, obst4] #list of obstacles
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
# Plot
fig, ax = plt.subplots()
plt.ylim((-15,15))
plt.xlim((-15,15))
ax.set_aspect('equal', adjustable='box')
pcur = tree.nodes[tree.pcurID, 0:2]
utils.drawShape(patches.Circle((pcur[0], pcur[1]), 0.5, facecolor = 'red' ), ax)
utils.drawTree(tree.nodes, ax, 'grey')
utils.drawPath(solPath, ax)
utils.plotEnv(tree, goal,start, ax)
im = utils.saveImFromFig(fig)
cv.imshow('frame',im)
cv.waitKey(1000)
plt.close()
####################
####################
#4. Init movement()-->> update pcurID 
solPath,solPathID = tree.nextSolNode(solPath,solPathID)
####################
#5. Begin replanning loop, while pcur is not goal, do...
while np.linalg.norm(tree.nodes[tree.pcurID, 0:2] - goal) > epsilon:
# for i in range(20):
	fig, ax = plt.subplots()
	plt.ylim((-15,15))
	plt.xlim((-15,15))
	ax.set_aspect('equal', adjustable='box')
	pcur = tree.nodes[tree.pcurID, 0:2]
	utils.drawShape(patches.Circle((pcur[0], pcur[1]), 0.5, facecolor = 'red' ), ax)
	utils.drawTree(tree.nodes, ax, 'grey')
	utils.drawPath(solPath, ax)
	utils.plotEnv(tree, goal,start, ax)
	im = utils.saveImFromFig(fig)
	cv.imshow('frame',im)
	cv.waitKey(500)
	plt.close()
	# cv2.imwrite("image_{}".format(i), im) 
	#6. Obstacle Updates
	# tree.updateObstacles()
	tree.updateObstacles()
	# tree.obstacles[0].position = np.array([ 7, 2 ])
	#7. if solPath breaks, replan
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
		reconnected,solPath,solPathID = tree.reconnect(separatePathID)

		if reconnected:
			print('\n')
			print("				RECONNECT SUCCESSFUL !		")
			print('\n')
			utils.drawTree(tree.nodes, ax, 'red')
		
		else:
			print('\n')
			print("				RECONNECT FAILED!		")
			print('\n')
			#21. if reconnect fails, regrow tree
			solPath,solPathID = tree.regrow()

			print('\n')
			print("				REGROW SUCCESSFUL !		")
			print('\n')
	######## END REPLANNING Block #######
	#26. Move to next sol node
	solPath,solPathID = tree.nextSolNode(solPath,solPathID)



plt.show()