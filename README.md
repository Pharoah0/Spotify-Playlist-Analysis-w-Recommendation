# Spotify Playlist Analysis with Song Recommendation
Author: Pharoah Evelyn  
<p align="center">
    <img src="https://github.com/Pharoah0/Spotify-Playlist-Analysis-w-Recommendation/blob/main/images/Spotify_Logo.jpeg" />
</p>

## Overview

#### This notebook aims to analyze & compare song features within my curated playlist & Spotify playlists, respectively, followed by building a model that can best predict song inclusion in my playlist.

#### We also explore different song recommendation options.

This notebook uncovers some of the backend details about what makes my playlist unique. My playlist has built up over the years, and I've only added songs based on specific criteria, such as being Lofi, having a nice groove, and, at the very least, songs that I can repeatedly listen to.

There are numerous ways to discover music for this ever-growing playlist, such as going on a song's radio or tuning into playlists that Spotify creates for you. 

In this repo, I employ the Spotify API to capture playlists, explore the quality of playlist suggestions, build predictors to see if songs are in my curated playlist, then use recommendation systems for individual songs, pulling from the recommended playlists.
## Business Problem

**Spotify** is a digital music, podcast, and video service that gives you access to millions of songs and other content from creators worldwide. Thus, content recommendation is vital for its success and maintaining user satisfaction.

In this case, Spotify seeks to classify songs to assign to curated playlists. Numerous users, such as myself, primarily listen to a small selection of music. This analysis will help users discover new music as they please, whether via recommendations from the home page or by enhancing one’s playlists. 

It will allow the recommendation algorithm to introduce new music to users via their homepage and improve their playlists & listening experience.
## Data Preparation
<p align="center">
    <img src="https://github.com/Pharoah0/Spotify-Playlist-Analysis-w-Recommendation/blob/main/images/Spotify_API_Quick_Look.jpeg" />
</p>

I used the Spotify API to capture all of the data presented in this repo. To do so, I needed to create an app on the [Spotify for Developers](https://developer.spotify.com/) webpage. There, numerous resources describe how to use the Spotify API. After creating an account, there are public & private keys to your app that we employed to connect to the Spotify API & pull data.

The data pulled in this case was my playlist: 'Instrumentals.'
<p align="center">
    <img src="https://github.com/Pharoah0/Spotify-Playlist-Analysis-w-Recommendation/blob/main/images/My_Playlist.png" width="90%" height="90%"/>
</p>

This playlist was saved as `my_playlist.csv` in the data folder in this repo.

I also used playlists based on my listening history, as well as what spotify says is more of what I like.
<p align="center">
    <img src="https://github.com/Pharoah0/Spotify-Playlist-Analysis-w-Recommendation/blob/main/images/Based_on_Recent_Listening.png" width="85%" height="85%"/>
</p>
<p align="center">
    <img src="https://github.com/Pharoah0/Spotify-Playlist-Analysis-w-Recommendation/blob/main/images/More_of_What_You_Like.png" width="85%" height="85%"/>
</p>

I saved all playlists in these images into the data folder, and you can view their contents there.

Once I imported the data, I performed EDA to ensure that this data was usable before employing the functions we needed to solve our business problem.

In this case, the data had no null values and a few instances of duplicate entries of songs. Therefore, EDA was minimal.
## Methods Used
Before EDA, I loaded in the datasets two different ways:
* Loaded in my playlist as a standalone dataframe
* Used the glob module to load all remaining playlists into one dataframe


Next, I looked at the distribution columns of the numerical features, a heat map to show the relationship of these features to each other, and displayed the statistics of these features: 

* **Length** — The length of a song measured in milliseconds
* **Popularity** — The popularity of the track. The value will be between 0 and 100, with 100 being the most popular.
* **Acoustiness** — A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence that the track is acoustic.
* **Danceability** — Danceability describes how suitable a track is for dancing based on musical elements, including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is the least danceable, and 1.0 is the most danceable.
* **Valence** — A measure from 0.0 to 1.0 describes the musical positiveness conveyed — tracks with high valence sound more positive.
* **Key** — The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation. 
  * Ex: 0 = C, 1 = C♯/D♭, 2 = D, and so on. If Spotify detected no key, the value is -1.
* **Energy** — Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.
* **Instrumentalness** — Predicts whether a track contains no vocals. The closer the instrumentals value is to 1.0, the greater likelihood the track has no vocal content.
* **Liveness**  — Detects the presence of an audience in the recording, ranging from 0 to 1.
* **Loudness** — The overall loudness of a track in decibels (dB). Values typical range between -60 and 0 dB.
* **Speechiness** — Probability of a song containing only speech. Spoken word tracks and vocal intros will have values close to 1.
* **Mode** — Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1, and minor is 0.
* **Tempo** — The overall estimated tempo of a track in beats per minute (BPM).
* **Time Signature** — A notational convention to specify how many beats are in each bar (or measure)


#### I discovered that in my playlist:
* Energy, instrumentalness, acousticness, & loudness have the most impact on being included in my playlist.
* Other key pairings I am noticing are valence & danceability, 
* Song length, speechiness, popularity & time signature appear to be the least relevant for the impact of features in this playlist.
  * With that said, I usually don't add songs with vocals (or verses) to this playlist. So I will keep speechiness as a critical marker.
* Energy and loudness, in particular, have the highest multicollinearity seen on this chart. This makes sense to me, as a song with high energy tends to have an increasing rhythmic complexity, whether melodically or in the background accompaniment with various instruments (as reflected in this chart as well.
Afterward, I compared the distributions between my playlist & my recommended songs dataframe

<p align="center">
    <img src="https://github.com/Pharoah0/Spotify-Playlist-Analysis-w-Recommendation/blob/main/images/Feature_Distribution_Comparisons.png" width="95%" height="75%"/>
</p>

Upon direct comparison of each feature in both datasets, we can see that they appear similar with their distributions, albeit with slight variations.
There is a higher distribution with songs of lower valence in the recommended songs dataset compared to my playlist. This suggests that I tend to like happier-sounding songs.
Popularity in my playlist also seems to be more varied, whereas a noticeable selection of songs in the second dataset are popular. In contrast, songs on my playlist don't necessarily cater to popular music.

Once I finished the initial analysis, I concatenated my playlist and the recommended songs dataframe into a new dataset. I gave each a target column value, 1 for positive (present in my playlist) and 0 for naught, binarizing the data.

Next, we ran a few classifier models and directly compared their metrics using a function.

Below are the results!
## Models
For ease of viewing, the top 2 models are displayed here. For more results, please see my accompanying jupyter notebook in this repository!

#### Logistic Regression
<img src="https://github.com/Pharoah0/Spotify-Playlist-Analysis-w-Recommendation/blob/main/images/LR_Model.png" width="50%" height="50%">

Interpretation of Playlist Inclusion with a Logistic Regression classifier:  
- 411 True Negatives: Songs not in my playlist  
- 141 False Positives: Predicted songs to be in my playlist, but are not present  
- 16 False Negatives: Predicted songs to not be in my playlist, but are present  
- 28 True Positives: Songs in my playlist  

#### SVM
<img src="https://github.com/Pharoah0/Spotify-Playlist-Analysis-w-Recommendation/blob/main/images/SVM_Model.png" width="50%" height="50%">

Interpretation of Playlist Inclusion with a Support Vector Machine classifier:  
- 447 True Negatives: Songs not in my playlist  
- 105 False Positives: Predicted songs to be in my playlist, but are not present  
- 18 False Negatives: Predicted songs to not be in my playlist, but are present  
- 26 True Positives: Songs in my playlist  

## Song Recommendation
<p align="center">
    <img src="https://github.com/Pharoah0/Spotify-Playlist-Analysis-w-Recommendation/blob/main/images/headphones.jpeg" width="95%" height="95%"/>
</p>

In my analyzing_playlists notebook, I  compared two different routes of content-based song recommendation.

One of which will use the cosine similarity from a matrix
& The other uses neighborhood collaborative filtering using the similarity metrics method

I am choosing this approach because the goal is to recommend specific songs that are similar to my favorite binge-worthy tunes.

## Conclusions
#### The best model is the Support Vector Machine model.

This is because: 
Based on both the individual model outputs and the direct comparisons, it app
ears, our SVM machine best predicted songs in my playlists and songs outside of the playlist.

I conclude because this classifier incorrectly predicts songs in my playlist less than logistic Regression and retains a similar correct prediction rate as the logistic Regression.

In this case, Decision Trees & Random Forests, despite accounting for class imbalances, seem to rely on automatically predicting for 0: songs not being in my playlist.

#### Song Recommender:
Though the two different recommenders return different results, there appears to be an agreement with some songs in their results.

Due to the songs having similar features throughout all variables, tracks that appear in both recommenders must be reliable matches to the music searched and might be a likely lead to be added to my playlist.

There are also some instances of recommended songs that are already present in my playlist. But that can be attributed to them being present in this dataset.
## Recommendations
#### Usage for our model & recommender systems:
* Use the best model to predict songs in other playlists
* Use recommenders in conjunction with each other to find music that's closely related to your favorite songs
* Spotify can employ different techniques at different times in the backend, running different models and recommender systems on a playlist level as well as on the song level; different approaches to bring users good music, nonetheless

#### Suggestions:
* To further increase model performance, we could implement a grid search on all models to discover if higher model performance is probable.
* We could also implement this on a smaller sample size of songs, thus will impact model performance if necessary
## Next Steps
* One can run a cumulative recommendation for every song in a given playlist
* Utilize Spotify deep audio analysis with audio features & neural networks
  * An analysis of this type will be based on the actual audio samples of songs
  * This will lead to a much deeper analysis & model building methodology
* Automate radio stations based on playlist or most played songs from the playlist 
## Repository Structure
A description of the structure of this repository and its contents:
```
└── images                          <- Both sourced externally and generated from code
└── data                            <- Sourced from using the Spotify API on my profile
└── analyzing_playlists.ipynb  
    ^^ Narrative documentation of analysis in Jupyter notebook
└── gathering_playlists.ipynb   
    ^^ Step-by-step process of utilizing the Spotify API and collecting data from my user account
└── Spotify-Playlist-Analysis-w-Recommendation.pdf       <- PDF version of project presentation
└── README.md                       <- The top-level README for reviewers of this project

```
