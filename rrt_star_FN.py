import numpy as np
from Obstacle import Obstacle
from Tree import Tree
import utils
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


#########################################
############# TASK SET UP ###############
#########################################
start = [0,0]
goal = [10,10]
xmin, ymin, xmax, ymax = -15,-15,15,15 #grid world borders
obst1 = Obstacle('rect',[-5, -5, 1,1], [0,0], np.eye(2))
obst2 = Obstacle('circle',[2,-2,1], [0,0], np.eye(2))
obstacles = [obst1, obst2] #list of obstacles
N = 500 #numner of iterations
epsilon = 1.0 #near goal tolerance
maxNumNodes = 200 #upper limit on tree size 
eta = 0.4 #max branch length
goalFound = False


#########################################
######### Begin Iterations   ############
#########################################
#1. Initialize Tree
tree = Tree(start, goal, obstacles)

for i in range(N):
	print("iter {} | Number of nodes: {}".format(i, np.shape(tree.nodes)[0]))
	#2. Sample
	# qrand = utils.sampleUniform(xmin,ymin, xmax, ymax)
	qrand = utils.sampleGaussian(goal, 20.0*np.eye(2))
	if goalFound:
		qrand = utils.sampleUniform(xmin,ymin, xmax, ymax)

	#3. Steer from qnear toward qrand
	qnearID, qnear = tree.getNearest(qrand)
	qnew = utils.steer(eta, qnear, qrand)

	#4. If qnew is near goal set it to goal and store its id
	if np.linalg.norm(qnew - goal) < epsilon and tree.isValidBranch(qnear, goal):
		qnew = goal
		goalFound = True
		#4.1 Append qnewID(goalID) to tree.goalIDs list
		newGoalID = np.shape(tree.nodes)[0]
		tree.addGoalID(int(newGoalID))
	
	
	if tree.isValidBranch(qnear, qnew):
		#5. Find nearest neighbors within hyperball
		gamma = 10.0 #param to set 
		n = np.shape(tree.nodes)[0] #number of nodes in tree
		radius = max(eta, np.sqrt(gamma*(np.log(n)/n)))
		NNids = tree.getNN(qnew, radius)
		#6. Choose qnew's best parent and insert qnew
		naysID = np.append(np.array([qnearID]),NNids)
		qparentID = tree.chooseParent(qnew, naysID)	
		qparent = tree.nodes[qparentID, :]
		branchCost = np.linalg.norm(qnew - qparent)
		qnewID = tree.addEdge(int(qparentID), qnew, branchCost)	
		#7. Rewire tree within the hyperball vicinity
		tree.rewire(qnewID, naysID)

		#7. Trim tree
		if np.shape(tree.nodes)[0] > maxNumNodes:
			tree.forcedRemove(qnewID, goal, goalFound)
	
	#8. Print out cost to goal if goal has been found
	if goalFound:
		goalID, costToGoal = tree.minGoalID()
		print("		cost to goal: {}".format(costToGoal))



######################
#####Draw tree #######
######################
def draw_edge(a, b, ax, color = 'blue'):
    path = Path([(a[0], a[1]), (b[0], b[1])], [Path.MOVETO, Path.LINETO])
    pathpatch = patches.PathPatch(path, facecolor='white', edgecolor= color)
    ax.add_patch(pathpatch)

#Plot final tree
fig, ax = plt.subplots()
plt.ylim((-15,15))
plt.xlim((-15,15))
ax.set_aspect('equal', adjustable='box')

rect = patches.Rectangle((-5,-5), 1, 1)
circ = patches.Circle((2, -2), 1 )
ax.add_patch(rect)
ax.add_patch(circ)

for i in range(np.shape(tree.nodes)[0]):
	if tree.parents[i] is not None:
		draw_edge(tree.nodes[i], tree.nodes[tree.parents[i]], ax)

plt.show()




