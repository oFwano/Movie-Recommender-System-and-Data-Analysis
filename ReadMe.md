# **ANALYSIS AND COMPUTATION ON MOVIES**
Created a recommender system for movies using the Movielens dataset.

# Required Libraries:
* scikit-learn
* statsmodels
* matplotlib
* pandas
* numpy
* scipy

# Movielens dataset analysis

**movielens-analysis.py :**

Input Arguements: NONE

Output:
  * plot images to directory plot
  * Prints ANOVA p-Value of genre vs ratings
  * Prints top 3 recommended movies for each popular genre.


To run the movielens analysis function simply use the following command:

`python3 movielens-analysis.py`

If the output images are not being saved to plot the user must first create the
directory plot.

# Movielens machine learning and knn recommendation system

**movielens-ml.py :**

Input Arguements: NONE

Output:
  * 1 plot image of rating frequency to directory plot
  * Prints accuracy of selected machine learning models.
  * Prints top 10 recommended movies using knn.


To run the movielens machine learning program simply use the following command:

`python3 movielens-ml.py`

If user wants to select a different movie to recommend change the variable
on line 32 "movie_title" to the desired movie.

If the output images are not being saved to plot the user must first create the
directory plot.

![imagepreview](/imgpreview.png)
