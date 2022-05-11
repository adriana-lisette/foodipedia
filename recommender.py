# -*- coding: utf-8 -*-
"""
Created on Tue May 10 19:03:23 2022

@author: Adriana Garcia
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

food = pd.read_csv('mexican_food (1).csv')

#se inicializa la clase para convertir los tags a vectores
tfidf = TfidfVectorizer()

#se crea la matrix con los vectores obtenidos de los tags
tagMatrix = tfidf.fit_transform(food['tags'])

#se usa la funcion linear_kernel para encontrar la similitud entre 2 vectores
simMatrix = linear_kernel(tagMatrix,tagMatrix)

#se mapean las comidas con el numero de index
mapping = pd.Series(food.index,index = food['name'])
mapping

def recommend_food(food_input):
    food_index = mapping[food_input]

    #simScore nos dice los valores de similitud entre platillos por indices que se especificaron al mapear
    simScore = list(enumerate(simMatrix[food_index]))
    #se ponen en orden
    simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
    # Obtenemos los primeros 3
    simScore = simScore[1:4]
    #Regresamos los indices de los platillos
    food_indices = [i[0] for i in simScore]
    
    
    return (food['name'].iloc[food_indices])


def getDishInfo(food_input):
    dish = food[(food['name']== food_input)]
    ingredients = dish['ingredients'].values[0]
    calories = dish['calories'].values[0]
    prepTime = dish['wait_time'].values[0]
    
    return ingredients, calories, prepTime

