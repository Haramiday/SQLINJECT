from django.urls import path
from .views import Prediction

urlpatterns = [
    path('predict/', Prediction.as_view(), name = 'prediction'),
]