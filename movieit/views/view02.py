from django.shortcuts import render, redirect
from mymovie.models import NoticeTab
from datetime import datetime
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.http.response import HttpResponseRedirect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import urllib.request
from bs4 import BeautifulSoup
import requests
from PIL import Image



def mainFunc(request):
    # 네이버영화 현재상영작 평점순으로 크롤링 
    '''
    url = "https://movie.naver.com/movie/running/current.naver?view=image&tab=normal&order=likeCount"
    soup = BeautifulSoup(urllib.request.urlopen(url).read(), 'html.parser')
    #content > div.article > div > div.lst_wrap > ul > li:nth-child(5) > a > strong
    #content > div.article > div > div.lst_wrap > ul > li:nth-child(4) > a > img
    title_list = []

    for tag in soup.select('div.lst_wrap > ul a > strong'):
        title_list.append(tag.text.strip())
        
    df = pd.DataFrame(title_list, columns=['제목'])
    print(df)
    
    return render(request, 'main.html', {'title':title_list})
    
    
    response = request
    movie_data = pd.read_csv('pypro3/movieit/movie_naver.csv', header=None)
    movie_data = pd.DataFrame(movie_data.iloc[0,:])
    print(movie_data.head(3))
    '''
    file = []
    for i in range(1,31):
        file.append('./static/image/Rank'+f'{i}'+'.png')
    print(file)
    
    title = []
    # data = pd.read_csv("C:/Users/jny/Desktop/GitRepository/Movieit/movieit/mymovie/static/movie_summary.csv", encoding='unicode_escape')
    # print(data.head(3))
    path = "D:/work/psou/movieit/mymovie/static/movie_summary.csv"
    data = pd.read_excel(path)
    title = []
    for t in data['영화제목']:
        title.append(t)
    print(title)
    
    return render(request, 'main.html', {'file':file, 'title':title})
        
