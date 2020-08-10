#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math

from scipy import stats

def main():
    ratings_path = "Data/ml-latest-small/ratings.csv"
    movies_path = "Data/ml-latest-small/movies.csv"

    #-- Extract and clean data for analysis --#
    df_movies = pd.read_csv(movies_path,usecols=['movieId','title','genres'])
    df_ratings = pd.read_csv(ratings_path,usecols=['userId','movieId','rating','timestamp'])


    #--------- Cleaning data for Plot Average Rating of Popular Genres per Year ---------#
    #-- 1. Convert user rating timestamp to year --#
    #-- 2. Divide movie into each labelled genre --#
    #-- 3. Calculate rating mean per year        --#

    df_ratings['date'] =  pd.to_datetime(df_ratings['timestamp'],unit='s')
    df_ratings['year_rated'] = df_ratings['date'].dt.year
    df_ratings = df_ratings.drop(columns=['timestamp','date'])

    df_movies['genres'] = df_movies['genres'].apply(lambda x: x.split("|"))
    df_movies['year_released']= df_movies['title'].str.extract(r'\((\d\d\d\d)\)').fillna(0)
    df_movies['year_released']= df_movies['year_released'].astype('int')
    df_merged = pd.merge(df_movies,df_ratings, on='movieId')
    df_merged = df_merged.explode('genres')

    df_plt = df_merged.groupby(['genres', 'year_rated']).agg({'rating': 'mean'})
    df_plt = df_plt.reset_index()
    df_plt = df_plt[df_plt['year_rated']!=0]
    df_plt3 = df_merged.groupby(['year_rated']).agg({'rating': 'count'})
    df_plt3 = df_plt3.reset_index()


    #--------- Cleaning data for counting movies per genre and calculate average ratiung ---------#    
    #-- 1. Calulate mean rating grouped by genres --#
    #-- 2. Count number of movies per genre       --#
    #-- 3. Used top genres to create variable 'genres' in this case it was with count > 10k --#

    df_grouped = df_merged[['genres','rating']].groupby('genres')
    df_mean = df_grouped.agg('mean')
    count = df_grouped['genres'].count()
    df_count = pd.DataFrame({'genres': count.index, 'movies': count.values})
    df_count_genres = pd.merge(df_mean, df_count, on='genres')
    df_plt2 = df_count_genres.sort_values(by='movies', ascending=False)


    #-- choosing genres to analyze --#

    #-- Genres with a movie count of <10k (listed from most to least) --#
    #genres = ['Children','Mystery', 'Horror','Animation','War', 'IMAX','Musical','Western','Documentary','Film-Noir']

    #-- Genres with a movie count of >10k (listed from most to least) --#
    genres = ['Drama','Comedy','Action','Thriller','Adventure','Romance','Sci-Fi','Crime','Fantasy']  
    top_genres_2013 = ['Crime','Drama','Thriller','Fantasy','Sci-Fi','Action']

    #-- calculating std and average rating per genre -- #
    df_genres = df_plt[df_plt['genres'].isin(set(genres))].drop(columns='year_rated')

    std_array = []
    mean_array = []
    for g in genres:
        data = df_genres[df_genres['genres']==g].std()
        data2 = df_genres[df_genres['genres']==g].mean()
        std_array.append(float(data))
        mean_array.append(float(data2))

    #-- cleaning data to find top 3 movies per genre in 2013 to further analyze plot 1 --#
    df_select_movies = df_merged[df_merged['genres'].isin(set(top_genres_2013))]
    df_select_movies = df_select_movies[df_select_movies['year_rated'] == 2013]
    df_select_movies = df_select_movies[df_select_movies['year_released'] == 2013]
    df_select_movies = df_select_movies[['title','genres','rating']].groupby(['title','genres']).agg('mean')
    df_select_movies = df_select_movies.sort_values(by=['genres','rating'], ascending=False).reset_index()


    
    
    #----------------------------- plots -----------------------------#
    #-- Plot 1 Average Rating of Popular Genres per Year --#
    fig = plt.figure(1, figsize=(20,10))
    for g in genres:
        data = df_plt[df_plt['genres'] == g]
        plt.plot(data.year_rated, data.rating, 'o-')
    plt.legend(genres, prop={'size': 15})
    plt.xlabel('Year',fontsize = 20)
    plt.ylabel('Average Rating',fontsize = 20)
    plt.title('Average Rating of Popular Genres per Year',fontsize = 30)
    plt.savefig('plot/Avg_genre_rating_per_year', fontsize=40,bbox_inches='tight')
    #plt.show()
    
    
    
    #-- Plot 2 Movies Per Genre --#
    fig = plt.figure(2, figsize=(25,10))
    plt.ylabel('Number of Movies', fontsize = 20)
    plt.xlabel('Genre', fontsize = 20)
    plt.bar(df_plt2.genres, df_plt2.movies)
    plt.title('Movies per Genre (All)',fontsize = 30)
    plt.savefig('plot/num_movies_per_genre', fontsize=35,bbox_inches='tight')
    #plt.show()
    
    
    #-- Plot 3 mean and std of movie genre --#
    fig = plt.figure(3, figsize=(20,15))
    plt.ylabel('Average Rating', fontsize = 20)
    plt.xlabel('Movie Genres', fontsize = 20)
    plt.bar(genres,mean_array,yerr=std_array)
    plt.title('Average Rating per Movie Genre (Popular)',fontsize = 30)
    plt.savefig('plot/movies_ratings_per_genre_popular', fontsize=35,bbox_inches='tight')
    #plt.show()

              
    #-- Plot 4 numbers of ratings per year --#
    fig = plt.figure(4, figsize=(20,15))
    plt.ylabel('Average Rating', fontsize = 20)
    plt.xlabel('Movie Genres', fontsize = 20)
    barlist = plt.bar(df_plt3.year_rated,df_plt3.rating)
    barlist[17].set_color('g')
    plt.title('Number of Ratings per year',fontsize = 30)
    plt.savefig('plot/ratings_per_year', fontsize=35,bbox_inches='tight')
    
    
    
    #-- final print statements --#
    print("ANOVA for ratings of all genres: ")
    print(stats.f_oneway(df_count_genres['rating'],df_count_genres['movies']), end="\n\n")
    print("With p < 0.05, there is a difference between the means of the movie ratings based on genres ")
    
    for g in top_genres_2013:
        data = df_select_movies[df_select_movies['genres'] == g]
        data = data.drop(columns='genres')
        print("\nTop 3 rated movies in the genre: {0}".format(g))
        print("-----------------------------------------")
        print(data.head(3).to_string(index=False))



if __name__ == '__main__':
    main()





