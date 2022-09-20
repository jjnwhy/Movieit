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
from django.urls.conf import include
from views import view01, view02, view1, view2
from myapp import views
 
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', view02.mainFunc),
    path('notice', view01.mainFunc),
    path('notice/', include('mymovie.urls')), # 위임하기
    path('input', views.inputFunc),
    path('input/', include('myapp.urls')),
    path('recommend_movie', views.recommend_movie),
    path('qna', view1.listFunc),
    path('qna/',include('myqna.urls')),
    path('movie',view01.detailFunc),
    path('movie/',include('mymovie.urls')),
] 
