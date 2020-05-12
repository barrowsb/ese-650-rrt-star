import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import io
import cv2 as cv

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
def draw_edge(a, b, ax, color = 'blue', width = 1):
    path = Path([(a[0], a[1]), (b[0], b[1])], [Path.MOVETO, Path.LINETO])
    pathpatch = patches.PathPatch(path, facecolor='white', edgecolor= color, linewidth = width)
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
		draw_edge(path[i], path[i+1], ax, color, 2)
def plotEnv(tree, ax):
	#draw obstacles
	for obs in tree.obstacles:
		drawShape(obs.toPatch('blue'), ax)
	#draw start and goal
	G = patches.Circle((tree.goal[0], tree.goal[1]), 0.5, facecolor = 'orange' )
	S = patches.Circle((tree.start[0], tree.start[1]), 0.5, facecolor = 'pink' )
	drawShape(G, ax)
	drawShape(S, ax)

def saveImFromFig(fig, dpi= 180):
	buf = io.BytesIO()
	fig.savefig(buf, format = "png", dpi = dpi)
	buf.seek(0)
	img_arr = np.frombuffer(buf.getvalue(), dtype = np.uint8)
	buf.close()
	img = cv.imdecode(img_arr,1)
	return img

def generate_plot(tree,solPath):
	fig,ax = plt.subplots()
	plt.ylim((-15,15))
	plt.xlim((-15,15))
	ax.set_aspect('equal',adjustable='box')
	pcur = tree.nodes[tree.pcurID,0:2]
	drawShape(patches.Circle((pcur[0],pcur[1]),0.5,facecolor = 'red'),ax)
	drawTree(tree.nodes,ax,'grey')
	drawPath(solPath,ax)
	plotEnv(tree,ax)
	im = saveImFromFig(fig)
	cv.imshow('frame',im)
	cv.waitKey(100)
	# Converting from BGR (OpenCV representation) to RGB (ImageIO representation)
	im = cv.cvtColor(im,cv.COLOR_BGR2RGB)
	plt.close()

	return im
######################################################################################