import numpy as np


def sampleUniform(xmin, ymin, xmax, ymax):
	return np.array([np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax) ])

def sampleGaussian(mean, covar):
	return np.random.multivariate_normal(mean, covar)

def steer(eta,qnear, qrand):
	dist = np.linalg.norm(qrand - qnear)
	branchLength = min(eta,dist)
	qdir = branchLength * (qrand - qnear)/dist
	return qnear + qdir