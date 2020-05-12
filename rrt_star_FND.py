import numpy as np
from Obstacle import Obstacle
from Tree import Tree
import utils
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import cv2 as cv
import imageio
import time


#########################################
############### Task Setup ##############
#########################################
start = [-12,-12]
goal = [11,11]
chaos = 0.05
xmin, ymin, xmax, ymax = -15,-15,15,15 #grid world borders
obst1 = Obstacle('rect',[-5, 5, 2,3], [0,0], chaos*np.eye(2), 1.5)
obst2 = Obstacle('circle',[3,9,2], [0,0], chaos*np.eye(2), 1.5)
obst3 = Obstacle('rect', [8,-5,1,6], [0,0], chaos*np.eye(2), 1.5)
obst4 = Obstacle('rect', [-3,-3,7,1], [0,0], chaos*np.eye(2), 1.5)
obst5 = Obstacle('circle', [-10,-6,2], [0,0], chaos*np.eye(2), 0.5)
obst6 = Obstacle('rect', [-12,9,2,2], [0,0], chaos*np.eye(2), 0)
obstacles = [obst1, obst2, obst3, obst4, obst5, obst6] #list of obstacles
epsilon = 0.5 #near goal tolerance
maxNumNodes = 3000 #upper limit on tree size 
eta = 1.0 #max branch length
gamma = 20.0 #param to set for radius of hyperball
goalFound = False

#########################################
# Creating a list to store images at each frame
images = []

#########################################
########### Begin Iterations ############
#########################################
#1. Initialize Tree and growth
print("Initializing FN TREE.....")
tree = Tree(start, goal, obstacles, xmin,ymin,xmax, ymax, maxNumNodes = maxNumNodes)

#2. Set pcurID = 0; by default in Tree instantiation

#3. Get Solution Path
solPath, solPathID = tree.initGrowth(exhaust = True, FN = True)

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
# Converting from BGR (OpenCV representation) to RGB (ImageIO representation)
im = cv.cvtColor(im,cv.COLOR_BGR2RGB)
# Appending to list of images
images.append(im)
cv.waitKey(100)
plt.close()
####################

#4. Init movement()-->> update pcurID 
solPath,solPathID,dt = tree.nextSolNode(solPath,solPathID)
startTime = time.time()
#5. Begin replanning loop, while pcur is not goal, do...
while np.linalg.norm(tree.nodes[tree.pcurID, 0:2] - goal) > epsilon:
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
	# Converting from BGR (OpenCV representation) to RGB (ImageIO representation)
	im = cv.cvtColor(im,cv.COLOR_BGR2RGB)
	# Appending to list of images
	images.append(im)
	cv.waitKey(100)
	plt.close()

	#6. Obstacle Updates
	tree.updateObstacles(dt)

	#7. if solPath breaks, replan
	if tree.detectCollision(solPath):

		#8. Stop Movement
		print("********************************************************")
		print("********* Path Breaks, collision detected! *************")
		print("********************************************************")

		#9. select remaining valid branches
		solPathID = tree.selectBranch(tree.pcurID, solPathID)	
		#10. Separate tree
		separatePathID, orphanedTree = tree.validPath(solPathID)
		separatePath = orphanedTree[separatePathID, 0:2].reshape(-1,2)

		#11-20. Try to reconnect main with orphanedTree
		reconnected,solPath,solPathID = tree.reconnect(separatePathID)

		if reconnected:
			print('\n')
			print("				RECONNECT SUCCESSFUL !		")
			print('\n')
			# utils.drawTree(tree.nodes, ax, 'red')
		
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
	solPath,solPathID,dt = tree.nextSolNode(solPath,solPathID)

print("Total Run Time: {} secs".format(time.time() -startTime))
costToGoal, goalID = tree.minGoalID()
print("Final Total Cost to Goal: {}".format(costToGoal))
plt.show()

# Closing the display window
cv.destroyAllWindows()

# Saving the list of images as a gif
print("The results are saved as a GIF to Animation_rrt_star_FND.gif")
imageio.mimsave('Animation_rrt_star_FND.gif',images,duration = 0.5)