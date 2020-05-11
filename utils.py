import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

def sampleUniform(xmin, ymin, xmax, ymax):
	return np.array([np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax) ])

def sampleGaussian(mean, covar):
	return np.random.multivariate_normal(mean, covar)

def sampleNearPath(path, covar = 1.0*np.eye(2)):
	#draws from Gaussian centered at randomly interpolated wayPt on path
	numWayPts = np.shape(path)[0]
	randID = np.random.randint(0, numWayPts-2)
	alpha = random.random()
	mean = alpha*path[randID] + (1-alpha)*path[randID+1] #interpolate
	return np.random.multivariate_normal(mean, covar)

def steer(eta,qnear, qrand):
	dist = np.linalg.norm(qrand - qnear)
	branchLength = min(eta,dist)
	qdir = branchLength * (qrand - qnear)/dist
	return qnear + qdir

######################################################################################
##############################
###### DRAWING METHODS #######
##############################
def draw_edge(a, b, ax, color = 'blue'):
    path = Path([(a[0], a[1]), (b[0], b[1])], [Path.MOVETO, Path.LINETO])
    pathpatch = patches.PathPatch(path, facecolor='white', edgecolor= color)
    ax.add_patch(pathpatch)
def drawTree(mat, ax, color = 'black'):
	#mat: nx4 matrix
	for i in range(np.shape(mat)[0]):
		if mat[i, 3] != -1:
			parentID = int(mat[i, 3])
			draw_edge(mat[i, 0:2], mat[parentID, 0:2], ax, color)
def drawShape(patch, ax):
	ax.add_patch(patch)
def drawPath(path, ax, color = 'green'):
	for i in range(np.shape(path)[0]-1):
		draw_edge(path[i], path[i+1], ax, color)
def plotEnv(tree, goal, start, ax):
	#draw obstacles
	for obs in tree.obstacles:
		drawShape(obs.toPatch('blue'), ax)
	#draw start and goal
	G = patches.Circle((goal[0], goal[1]), 0.5, facecolor = 'orange' )
	S = patches.Circle((start[0], start[1]), 0.5, facecolor = 'pink' )
	drawShape(G, ax)
	drawShape(S, ax)

def saveImFromFig(fig, dpi= 180):
	buf = io.BytesIO()
	fig.savefig(buf, format = "png", dpi = dpi)
	buf.seek(0)
	img_arr = np.frombuffer(buf.getvalue(), dtype = np.uint8)
	buf.close()
	img = cv2.imdecode(img_arr,1)
	return img
######################################################################################