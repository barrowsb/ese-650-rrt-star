# An Analysis of RRT\*, RRT\*FN, and RRT\*FND in a Dynamic Environment
##### Brian Barrows, Karan Pandya, and Sottithat Winyarat
##### ESE-650: Learning in Robotics
##### Spring 2020 Final Project

## Running our code

The files of interest are **rrt_star.py**, **rrt_star_FN.py**, and **rrt_star-FND.py**. Running each of these will run the corresponding algorithm in a 2D, dynamic environment with randomly moving obstacles, show the progress in real time, and output the resulting animation to the same directory as a GIF. The goal is shown in yellow, the robot in red, start location in pink, obstacles in blue, tree in gray, and best path in green. Additionally, the code will output the final cost (total distance traversed by the robot) and the total runtime of the script.

To run without visualization (as is done for the experiments in the *Results* section of our paper), simply changed the value of *plot_and_save_gif* in the *Task Setup* section of the script to *False*.

In some rare cases, the goal will not be reachable. This could happen if the robot or goal become totally encircled or due to bugs in our obstacle rebounding logic. Given the random nature of the environment, we were not able to robustly test for all cases. If our implementation fails to find the goal after adding 100,000 nodes to the tree, for whatever reason, the script will terminate.

## File breakdown

The bulk of the code is in the *Tree* class and the *Obstacle* class, in *Tree.py* and *Obstacle.py* respectively.

The *Tree* class handles all tree operations including growth (sampling, steering, connecting, rewiring), collision detection, cost propogation, forced node deletion, branch removal, rerooting, reconnecting, and regrowing. All methods required for obstacle-free RRT\*, RRT\*FN, and RRT\*FND are included here.

The *Obstacle* class handles the motion of obstacles (random changes in direction and rebounding), obstacle-level collision detection, and plotting.

The file *utils.py* contains various helper functions for sampling, steering, and plotting.

All files in the *./old* directory are deprecated versions and all files in the *./test* directory were for validating *Tree()* and *Obstacle()* class methods. In order to run these test files, you must have a copy of Tree.py and Obstacle.py in the same working directory.

## Results

**Example animations are included in the *./results* directory.**

All other results are included in that paper.
