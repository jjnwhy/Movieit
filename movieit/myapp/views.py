from django.shortcuts import render
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def mainFunc(request):
    return render(request, 'main.html')

def inputFunc(request):
    return render(request, 'show.html')

def recommend_movie(request):
    #로컬 경로
    # movie_data = pd.read_csv('pypro3/movieit/movie_naver.csv') 
    #깃헙 경로
    movie_data = pd.read_csv('https://raw.githubusercontent.com/jjnwhy/Movieit/feature_sm/movies%ED%8C%8C%EC%9D%BC/movie_naver.csv')


    KIM = int(request.POST.get('KIM'))
    NOPE = int(request.POST.get('NOPE'))
    LIMIT = int(request.POST.get('LIMIT'))
    BULLET = int(request.POST.get('BULLET'))
    EMERGE = int(request.POST.get('EMERGE'))
    WILOVE = int(request.POST.get('WILOVE'))
    SEOUL = int(request.POST.get('SEOUL'))
    ERROR = int(request.POST.get('ERROR'))
    ALIEN = int(request.POST.get('ALIEN'))
    LOTTO = int(request.POST.get('LOTTO'))
    CARTER = int(request.POST.get('CARTER'))
    TOP = int(request.POST.get('TOP'))
    HAN = int(request.POST.get('HAN'))
    HUNT = int(request.POST.get('HUNT'))
    LEAVE = int(request.POST.get('LEAVE'))
    
    # 새로 받은 데이터 가져오기
    new_data = np.array([[KIM,NOPE,LIMIT,BULLET,EMERGE,WILOVE,SEOUL,
                          ERROR,ALIEN,LOTTO,CARTER,TOP,HAN,HUNT,LEAVE]])
    new_data = pd.DataFrame(new_data, columns=['김호중 컴백 무비 빛이 나는 사람 PART 1. 다시 당신 곁으로', '놉', '리미트', '불릿 트레인', '비상선언',
       '사랑할 땐 누구나 최악이 된다', '서울대작전', '시맨틱 에러: 더 무비', '외계+인 1부', '육사오(6/45)',
       '카터', '탑건: 매버릭', '한산: 용의 출현', '헌트', '헤어질 결심'])
    # 기존 csv에 concat 하기
    print(new_data)
    con_data = pd.concat([movie_data, new_data], sort=False, ignore_index=True)
    
    print(con_data)
    # concat 된 데이터에 SVD 값 입력
    matrix = con_data.to_numpy()
    user_ratings_mean = np.mean(matrix, axis=1)
    matrix_user_mean = matrix - user_ratings_mean.reshape(-1,1)
    U, sigma, Vt = svds(matrix_user_mean, k = 10)
    sigma = np.diag(sigma)
    svd_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    df_svd_preds = pd.DataFrame(svd_user_predicted_ratings, columns = con_data.columns)
    print(df_svd_preds)
    print(new_data.values.reshape(-1))
    # 영화 추천

    # new_data 데이터 프레임으로 가져오기
    u_data={'영화제목': ['김호중 컴백 무비 빛이 나는 사람 PART 1. 다시 당신 곁으로', '놉', '리미트', '불릿 트레인', '비상선언','사랑할 땐 누구나 최악이 된다', '서울대작전', '시맨틱 에러: 더 무비', '외계+인 1부', '육사오(6/45)','카터', '탑건: 매버릭', '한산: 용의 출현', '헌트', '헤어질 결심'],
            '평점': new_data.values.reshape(-1)}
    user_data = pd.DataFrame(u_data)
    print(user_data)
    
    # new_data에서 SVD값 입력한 데이터 가져오기
    sorted_user={'영화제목': ['김호중 컴백 무비 빛이 나는 사람 PART 1. 다시 당신 곁으로', '놉', '리미트', '불릿 트레인', '비상선언','사랑할 땐 누구나 최악이 된다', '서울대작전', '시맨틱 에러: 더 무비', '외계+인 1부', '육사오(6/45)','카터', '탑건: 매버릭', '한산: 용의 출현', '헌트', '헤어질 결심'],
                 '평점': df_svd_preds.iloc[-1].values.reshape(-1)}
    sorted_user=pd.DataFrame(sorted_user)
    print('sorted ', sorted_user)
    
    # new_data에서 평점0점 준 영화만 가져오기
    recommend=user_data[user_data['평점'] == 0]
    print('rec1',recommend)
    # new_data에서 평점0점 준 영화들의 SVD 값 
    recommend=recommend.merge(sorted_user.reset_index(), on ='영화제목')
    print('rec2',recommend)
    # 영화 제목과 SVD 값 출력 (평점_y에 SVD 값이 입력됨)
    recommend=recommend[['영화제목','평점_y','index']]
    print('rec3', recommend)
    # 예측한 평점 상위 순으로 정렬
    recommend=recommend.sort_values(by=['평점_y'],ascending = False)
    print(recommend)
    # 추천 영화의 인덱스를 원래의 index 로 재설정
    recommend=recommend.set_index('index')
    print(recommend)
    # 3개만 추리기
    title=recommend.iloc[:3,0].to_dict

    context = {'title':title}

    return render(request, 'list.html', context)



    
    
    


