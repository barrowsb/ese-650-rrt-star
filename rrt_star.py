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
start = [-14,-14]
goal = [14,14]
chaos = 0.05
xmin, ymin, xmax, ymax = -15,-15,15,15 #grid world borders
obst1 = Obstacle('rect',[-5, 5, 2,3], [-0.8,-0.5], chaos*np.eye(2))
obst2 = Obstacle('circle',[3,9,2], [-0.5,0.5], chaos*np.eye(2))
obst3 = Obstacle('rect', [5,-2,1,4], [0,0], chaos*np.eye(2))
obst4 = Obstacle('rect', [-3,-3,7,1], [0,0], chaos*np.eye(2))
obstacles = [obst1, obst2, obst3, obst4] #list of obstacles
epsilon = 0.5 #near goal tolerance
eta = 1.0 #max branch length
gamma = 20.0 #param to set for radius of hyperball
goalFound = False

# Creating a list to store images at each frame
images = []

#########################################
########### Begin Iterations ############
#########################################
startTime = time.time()

#1. Initialize Tree and growth
print("Initializing RRT* TREE.....")
tree = Tree(start, goal, obstacles, xmin,ymin,xmax, ymax)

#2. Set pcurID = 0; by default in Tree instantiation

#3. Get Solution Path
solPath, solPathID = tree.initGrowth(exhaust = True, FN = False)

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
utils.plotEnv(tree, goal, start, ax)
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
solPath,solPathID = tree.nextSolNode(solPath,solPathID)

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
	tree.updateObstacles()

	#7. if solPath breaks, reset tree and replan
	if tree.detectCollision(solPath):
		print("********************************************************")
		print("**** Path Breaks, collision detected, Replanning! ******")
		print("********************************************************")
		tree.reset(inheritCost = True)
		solPath, solPathID = tree.initGrowth(exhaust = False, FN = False)

	######## END REPLANNING Block #######
	solPath,solPathID = tree.nextSolNode(solPath,solPathID)

print("Total Run Time: {} secs".format(time.time() - startTime))
costToGoal, goalID = tree.minGoalID()
print("Final Total Cost to Goal: {}".format(costToGoal))

# Closing the display window
cv.destroyAllWindows()

# Saving the list of images as a gif
print("The results are saved as a GIF to Animation_rrt_star.gif")
imageio.mimsave('Animation_rrt_star.gif',images,duration = 0.5)

