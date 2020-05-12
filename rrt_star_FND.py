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
goal = [12.,12.]
epsilon = 0.5 #near goal tolerance
goalLoc = [goal[0],goal[1],epsilon]
chaos = 0.05
xmin, ymin, xmax, ymax = -15,-15,15,15 #grid world borders
obst1 = Obstacle('rect',[-5, 5, 2,3], [0,0], chaos*np.eye(2), 1.5, goalLoc = goalLoc)
obst2 = Obstacle('circle',[3,9,2], [0,0], chaos*np.eye(2), 1.5, goalLoc = goalLoc)
obst3 = Obstacle('rect', [8,-5,1,6], [0,0], chaos*np.eye(2), 1.5, goalLoc = goalLoc)
obst4 = Obstacle('rect', [-3,-3,7,1], [0,0], chaos*np.eye(2), 1.5, goalLoc = goalLoc)
obst5 = Obstacle('circle', [-10,-6,2], [0,0], chaos*np.eye(2), 0.5, goalLoc = goalLoc)
obst6 = Obstacle('rect', [-12,9,2,2], [0,0], chaos*np.eye(2), 0, goalLoc = goalLoc)
obstacles = [obst1, obst2, obst3, obst4, obst5, obst6] #list of obstacles
maxNumNodes = 3000 #upper limit on tree size 
eta = 1.0 #max branch length
gamma = 20.0 #param to set for radius of hyperball
resolution = 0.0001
goalFound = False
plot_and_save_gif = True

# Creating a list to store images at each frame
if plot_and_save_gif:
	images = []

#########################################
########### Begin Iterations ############
#########################################
startTime = time.time()

#1. Initialize Tree and growth
print("Initializing Fixed Node Tree.....")
tree = Tree(start,goal,obstacles,xmin,ymin,xmax,ymax,maxNumNodes,resolution,eta,gamma,epsilon)

#2. Set pcurID = 0; by default in Tree instantiation

#3. Get Solution Path
solPath,solPathID = tree.initGrowth(exhaust = True,FN = True)

####################
# Plot
if plot_and_save_gif:
	im = utils.generate_plot(tree,solPath)
	# Appending to list of images
	images.append(im)
####################

#4. Init movement()-->> update pcurID 
solPath,solPathID,dt = tree.nextSolNode(solPath,solPathID)

#5. Begin replanning loop, while pcur is not goal, do...
while np.linalg.norm(tree.nodes[tree.pcurID,0:2] - tree.goal) > tree.epsilon:
	if plot_and_save_gif:
		im = utils.generate_plot(tree,solPath)
		# Appending to list of images
		images.append(im)

	#6. Obstacle Updates
	tree.updateObstacles(dt)

	#7. if solPath breaks, replan
	if tree.detectCollision(solPath):

		#8. Stop Movement
		print("********************************************************")
		print("********* Path Breaks, collision detected! *************")
		print("********************************************************")

		#9. select remaining valid branches
		solPathID = tree.selectBranch(solPathID)

		#10. Separate tree
		separatePathID, orphanedTree = tree.validPath(solPathID)

		#11-20. Try to reconnect main with orphanedTree
		reconnected,solPath,solPathID = tree.reconnect(separatePathID)

		if reconnected:
			print('\n')
			print("				RECONNECT SUCCESSFUL !		")
			print('\n')
		
		else:
			print('\n')
			print("				RECONNECT FAILED!		")
			print('\n')

			#21. if reconnect fails, regrow tree
			solPath,solPathID = tree.regrow()
			if solPath is None:
				print("Algorithm terminated ! \nUnable to connect to Goal even after 5000 attempts to regrow ! \n")
				break

			print('\n')
			print("				REGROW SUCCESSFUL !		")
			print('\n')

	######## END REPLANNING Block #######

	#26. Move to next sol node
	solPath,solPathID,dt = tree.nextSolNode(solPath,solPathID)


print("Total Run Time: {} secs".format(time.time() - startTime))

if solPath is not None:
	costToGoal, goalID = tree.minGoalID()
	print("Final Total Cost to Goal: {}".format(costToGoal))
	

if plot_and_save_gif:
	# Closing the display window
	cv.destroyAllWindows()

	# Saving the list of images as a gif
	print("The results are saved as a GIF to Animation_rrt_star_FND.gif")
	imageio.mimsave('Animation_rrt_star_FND.gif',images,duration = 0.5)