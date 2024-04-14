# Machine Learning
machine learning using python

Will be using the 3 Recommendation Systems
- Popularity Based - General overall popular 
---
    WR = (v / (v+m))*R + (m / (v+m))*C
    - where,
        - v = vote_count
        - m = minimum number of votes required at %
        - R = vote_average
        - C = average ratings for all movies
- Collaborative Filtering - After logging-in and base on a persons view with overall views, recommendations can be made.
- Content Based Filtering - When previewing an item, recommendations can be made from the preview and using the overall views.
---
    - frequency and uniqueness of a word gets a score 0.0-1.0
        - where, 0.0 - no relevance, 1.0 most relevance
    - next compare each a contents words with others to find similarities.
    - the similar words will give a score from 0.0-1.0
        - where, 0.0 - no relevance, 1.0 most relevance/similarities
    - highest score has similar content


Using [deepnote.com](https://deepnote.com/workspace/machinelearning-5530-2d0ce9d6-cbca-4486-87a1-4ae833dfa01f/project/Movie-Recommendations-System-36c77322-9d0e-4723-b7b4-890c883923b2/notebook/Collaborative-Base%20Filtering-a50dd223622f4ef1b618621e159915cc) as a jupyter notebook for machine learning