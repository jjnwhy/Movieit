from django.shortcuts import render
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# def mainFunc(request):
#     return render(request, 'main.html')

def inputFunc(request):
    return render(request, 'show.html')

def sentimentFunc(request):
    return render(request, 'sentiment.html')


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
    new_data = pd.DataFrame(new_data, columns=movie_data.columns)
    #columns=['김호중 컴백 무비 빛이 나는 사람 PART 1. 다시 당신 곁으로', '놉', '리미트', '불릿 트레인', '비상선언',
    #    '사랑할 땐 누구나 최악이 된다', '서울대작전', '시맨틱 에러: 더 무비', '외계+인 1부', '육사오(6/45)',
    #    '카터', '탑건: 매버릭', '한산: 용의 출현', '헌트', '헤어질 결심']
    # 기존 csv에 concat 하기
    print(new_data)
    con_data = pd.concat([movie_data, new_data], sort=False, ignore_index=True)
    
    print(con_data)
    # concat 된 데이터에 SVD 값 입력
    matrix = con_data.to_numpy()
    user_ratings_mean = np.mean(matrix, axis=1)
    matrix_user_mean = matrix - user_ratings_mean.reshape(-1,1)
    U, sigma, Vt = svds(matrix_user_mean, k = 13)
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
    recommend=recommend.iloc[:3,0]
    title=recommend.to_dict
    
    # index값을 이미지 파일명으로 쓰기 위한 경로 보내기
    img_path="/static/images/"
    
    context = {'title':title,'path':img_path}

    return render(request, 'list.html', context)


    
def sentiment_predict(request):
    REVIEW = str(request.POST.get("REVIEW", ""))
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    tokenizer = Tokenizer() 
    okt = Okt()
    # loaded_model = load_model("/movieit/best_model.h5")
    loaded_model = load_model("C:/Users/wnstm/OneDrive/바탕 화면/팀플/무브잇/Movieit/movieit/best_model.h5")
    REVIEW = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', REVIEW)
    REVIEW = okt.morphs(REVIEW, stem=True) # 토큰화
    REVIEW = [word for word in REVIEW if not word in stopwords] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([REVIEW]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = 30) # 패딩
    score = float(loaded_model.predict(pad_new)) # 예측
    star_score = round(score*10)
    
    PRED = star_score
    
    context = {'PRED' : PRED}
    
    return render(request, 'score.html', context)

def cosine_recommend_movie(request):
    movie_data = pd.read_csv('teampro/myapp/movie_naver.csv')

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
    new_data = np.array([[KIM, NOPE, LIMIT, BULLET, EMERGE, WILOVE, SEOUL,
                          ERROR, ALIEN, LOTTO, CARTER, TOP, HAN, HUNT, LEAVE]])
    # new_data = pd.DataFrame(new_data, columns=['김호중 컴백 무비 빛이 나는 사람 PART 1. 다시 당신 곁으로', '놉', '리미트', '불릿 트레인', '비상선언',
    #    '사랑할 땐 누구나 최악이 된다', '서울대작전', '시맨틱 에러: 더 무비', '외계+인 1부', '육사오(6/45)',
    #    '카터', '탑건: 매버릭', '한산: 용의 출현', '헌트', '헤어질 결심'])
    new_data = pd.DataFrame(new_data, columns=movie_data.columns)
    # 기존 csv에 concat 하기
    print(new_data)
    con_data = pd.concat([movie_data, new_data], sort=False, ignore_index=True)
    
    print(con_data)
    # concat 된 데이터에 SVD 값 입력
    matrix = con_data.to_numpy()
    user_ratings_mean = np.mean(matrix, axis=1)
    matrix_user_mean = matrix - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(matrix_user_mean, k=10)
    sigma = np.diag(sigma)
    svd_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    df_svd_preds = pd.DataFrame(svd_user_predicted_ratings, columns=con_data.columns)
    print(df_svd_preds)
    print(new_data.values.reshape(-1))
    
    corr = np.corrcoef(df_svd_preds)
    print(corr)
    
    # 새로운 사용자와 코사인 유사도. 제일 비슷한 회원은?
    user_sim = cosine_similarity(df_svd_preds, df_svd_preds)
    user_sim = pd.DataFrame(user_sim)
    print(user_sim.columns)
    print('user_sim : ', user_sim)
    print('user_sim : ', user_sim.sort_values(by=[133], axis=0 , ascending=False)[:5])
    top5_sim =  user_sim.sort_values(by=[133], axis=0 , ascending=False)[:5]
    # 회원 유사도
    sim = user_sim.sort_values(by=[133], axis=0 , ascending=False)[[133]][1:2]
    #sim = 새로운 사용자와 가장 유사한 사용자
    print('sim == ', sim)
    print(sim.values)
    print("= " * 30)
    # sim=sim.T
    coss = {'coss':sim.values}
    # print(sim.to_dict()[133])
    cos_sim = sim.to_dict()[133]
    pd.set_option('display.precision', 3)
    cos_sim = {'번 회원과의 유사도 ':cos_sim}
    sim_w_user = pd.DataFrame(cos_sim).join(movie_data)
    # print(pd.DataFrame(cos_sim).join(movie_data))

    # print('sim_w_user = ', sim_w_user)
    s = sim_w_user[sim_w_user != 0]
    s = s.dropna(axis=1)
    s = s.T
    # s = s.style.set_precision(3)
    # s = 새로운 사용자와 제일 유사한 사람 + 그 사람이 평가한 영화
    print()
    print('s : ', s)
    fig = plt.gcf()
    corr = np.corrcoef(corr)
    corr = corr[:10, :10]
    # sns.heatmap(corr, annot=True)
    sns.heatmap(corr, annot=True)
    # fig.savefig('../myapp/static/images/corr.png')
    
    
    sim_w_user = {'sim_w_user':s}
    
    # 상위 3개만 출력
    context = {'sim_w_user': sim_w_user}
    
    
    return render(request, 'show2.html', context)

"""
모델 만드는 부분

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

train_data = pd.read_csv('영화분류 4~10.csv')
# print('훈련용 리뷰 개수 :',len(train_data)) # 684

train_data['label'].value_counts().plot(kind = 'bar')
plt.show()  # 0 : 부정 / 1 : 긍정
# print(train_data.groupby('label').size().reset_index(name = 'count'))

# 한글과 공백을 제외하고 모두 제거
train_data['리뷰'] = train_data['리뷰'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# print(train_data['리뷰'][:5])
train_data['리뷰'] = train_data['리뷰'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
train_data['리뷰'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how = 'any')

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()

X_train = []
for sentence in tqdm(train_data['리뷰']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_train.append(stopwords_removed_sentence)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
# print(tokenizer.word_index) #2221개 단어


threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 횟수가 3보다 작으면 삭제
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

# print('단어 집합(vocabulary)의 크기 :',total_cnt)
# print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
# print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
# print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# 단어 집합(vocabulary)의 크기 : 2221
# 등장 빈도가 2번 이하인 희귀 단어의 수: 1592
# 단어 집합에서 희귀 단어의 비율: 71.67
# 전체 등장 빈도에서 희귀 단어 등장 빈도 비율: 22.93

vocab_size = total_cnt - rare_cnt + 1
# print('단어 집합의 크기 :',vocab_size) # 630

tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
# print(X_train[:3])

y_train = np.array(train_data['label'])

drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
# 빈 샘플들을 제거
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)

# 리뷰 길이 분포 시각화
plt.hist([len(review) for review in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

X_train = pad_sequences(X_train, maxlen=30)
embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)
loaded_model = load_model('best_model.h5')

"""