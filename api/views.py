import numpy as np
import pandas as pd
from .apps import ApiConfig
from rest_framework.views import APIView
from rest_framework.response import Response
#import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer




df = ApiConfig.CLEANED_DATA

#CountVectorizer
vectorizer = CountVectorizer(max_features=50, dtype = np.float32)
X = vectorizer.fit_transform(df["Sentence"].values).toarray()


class Prediction(APIView):
    def post(self, request):
        data = request.data
        value = data['sentence']
        #print(value)
        
        #vectorize
        sentence_vector = vectorizer.transform([value]).toarray()
        #get the sentence and reshape it
        model = ApiConfig.model
        predicted = model.predict(sentence_vector)[0]
        #print(predicted)
        if predicted == '0':
        	result = "normal"
        else:
        	result = "SQL Injection"
        response_dict = {"result": result}
        return Response(response_dict, status=200)