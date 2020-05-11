# An Analysis of RRT\*, RRT\*FN, and RRT\*FND in a Dynamic Environment
##### Brian Barrows, Karan Pandya, and Sottithat Winyarat
##### ESE-650: Learning in Robotics
##### Spring 2020 Final Project

## Running our code
The main classes are the *Tree()* class and the *Obstacle()* class, in *Tree.py* and *Obstacle.py* respectively. The *Tree()* class handles all tree operations included growth (sample, steering, connecting, rewiring), collision detection, cost propogation, forced deletion, branch removal, reroot, reconnect, and regrow. All methods required for obstacle-free RRT\*, RRT\*FN, and RRT\*FND are included here. The *Obstacle()* class handles the motion (random changes in velocity direction and rebounding), obstacle-level collision detection, and plotting.

The file *utils.py* contains various helper functions for sampling, steering, and plotting. All files in the "*/old/*" directory are deprecated versions and all files in the "*/test/*" directory were for validating *Tree()* and *Obstacle()* class methods.

The files of interest are **rrt_star.py**, **rrt_star_FN.py**, and **rrt_star-FND.py**. Running each of these will run the algorithms in a dynamic environment and show results.
