# CS205-KNN

## Overview

- This project is done for Project 2 of CS205
- This project includes the implementation of Forward Search Algorithm and Backward Search Algorithm
- It outputs the best set for predicting the result

## File Desciption

- feature_selection.cpp
  - Forward selection, Backward selection.
- knn_utils.cpp
  - Helper functions for feature selection.
- plot_utils.cpp
  - Helper funtions for drawing plots.
- main.cpp
  - Driver file.

## How To Execute

`cd part1`

`g++ -std=c++11 -o feature_selection_app main.cpp feature_selector.cpp knn_utils.cpp plot_utils.cppy`

## Performance Comparison

- Part 1
  ![Alt text](part1/forward_selection_results.png)
  ![Alt text](part1/backward_elimination_results.png)
- Part 2
  ![Alt text](part1/backward_elimination_results.png)
