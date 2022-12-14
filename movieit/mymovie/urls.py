"""movieit URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from views import view01

urlpatterns = [
    path('main', view01.mainFunc),
    
    path('insert', view01.insertFunc),
    path('insertok', view01.insertokFunc),
    
    path('search', view01.searchFunc), 
    
    path('content', view01.contentFunc),
    path('contentok', view01.contentokFunc),
    
    
    path('update', view01.updateFunc),
    path('updateok', view01.updateokFunc),
    
    path('delete', view01.deleteFunc),
    path('deleteok', view01.deleteokFunc),
    path('detail',view01.detailFunc),

    
]
