import numpy as np
import random


def sampleUniform(xmin, ymin, xmax, ymax):
	return np.array([np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax) ])

def sampleGaussian(mean, covar):
	return np.random.multivariate_normal(mean, covar)

def sampleNearPath(path, covar = 2.0*np.eye(2)):
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