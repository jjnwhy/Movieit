from datetime import datetime
from django.core.paginator import PageNotAnInteger, EmptyPage, Paginator
from django.http.response import HttpResponseRedirect
from django.shortcuts import render
from myqna.models import MyqnaQna

def mainFunc(request):
    aa="<div><h2>메인</h2></div>"
    return render(request,'main.html',{'main':aa})
def listFunc(request):
    # 댓글 처리
    data_all=MyqnaQna.objects.all().order_by('-gnum','onum')
    # 글 5개씩 보기
    per_page=5
    paginator=Paginator(data_all,per_page)
    page=request.GET.get('page')
    try:
        datas=paginator.page(page)
    except PageNotAnInteger: 
        datas=paginator.page(1)
    except EmptyPage: 
        datas=paginator.page(paginator.num_pages)
    return render(request,'board.html',{'datas':datas})
# 글 작성
def insertFunc(request):
    return render(request,'insert.html')
# 글 작성
def insertokFunc(request):
    if request.method=='POST':
        print(request.POST.get('name'))
        try:
            gbun=1 #Group number구하기
            datas=MyqnaQna.objects.all()
            if datas.count() !=0:
                gbun=MyqnaQna.objects.latest('id').id+1
            MyqnaQna(
                name=request.POST.get('name'),
                passwd=request.POST.get('passwd'),
                mail=request.POST.get('mail'),
                title=request.POST.get('title'),
                cont=request.POST.get('cont'),
                bip=request.META['REMOTE_ADDR'], #request.META.remote('REMOTE_ADDR')
                bdate=datetime.now(),
                readcnt=0,
                gnum=gbun,
                onum=0,
                nested=0,
            ).save()
        except Exception as e:
            print('추가 에러: ',e)
            return render(request,'error.html')
    #추가 후 목록으로
    return HttpResponseRedirect('/qna/list') 
    # return redirect('/board/list')
# Qna 검색기능 관련
def searchFunc(request):
    if request.method=='POST':
        s_type=request.POST.get('s_type')
        s_value=request.POST.get('s_value')
        print(s_type,s_value)
        #SQL의 like 연산과 유사한 칼럼명 __contains=value
        if s_type=='title':
            datas_search=MyqnaQna.objects.filter(title__contains=s_value).order_by('-id')
        elif s_type=='name':
            datas_search=MyqnaQna.objects.filter(name__contains=s_value).order_by('-id')
        per_page=5
        paginator=Paginator(datas_search,per_page)
        page=request.GET.get('page')
        try:
            datas=paginator.page(page)
        except PageNotAnInteger: #page의 값이 숫자가 아님
            datas=paginator.page(1)
        except EmptyPage: #page의 값이 비어있음
            datas=paginator.page(paginator.num_pages)

        return render(request,'board.html',{'datas':datas})
# Qna글 조회하기
def contentFunc(request):
    page=request.GET.get('page')
    data=MyqnaQna.objects.get(id=request.GET.get('id'))
    # 조회수 갱신
    data.readcnt=data.readcnt+1
    data.save() #update
   
    return render(request,'content.html',{'data_one':data,'page':page})
# Qna글 수정하기
def updateFunc(request):
    try:
        data=MyqnaQna.objects.get(id=request.GET.get('id'))
    except Exception as e:
        print('수정자료 읽기 오류:',e)
        return render(request,'error.html')
    
    return render(request,'update.html',{'data_one':data})
    
def updateokFunc(request):
    try:
        upRec=MyqnaQna.objects.get(id=request.POST.get('id'))
        #비밀번호 비교
        if upRec.passwd==request.POST.get('up_passwd'):
            upRec.name=request.POST.get('name')
            upRec.mail=request.POST.get('mail')
            upRec.title=request.POST.get('title')
            upRec.cont=request.POST.get('cont')
            upRec.save() #update
        else:
            return render(request,'update.html',{'data_one':upRec,'msg':'비밀번호 불일치'})
    except Exception as e:
        print('수정처리 오류:',e)
        return render(request,'error.html')
    
    return HttpResponseRedirect('/qna/list')
    
def deleteFunc(request):
    try:
        delData=MyqnaQna.objects.get(id=request.GET.get('id'))
    except Exception as e:
        print('삭제자료 읽기 오류:',e)
        return render(request,'error.html')
    
    return render(request,'delete.html',{'data_one':delData})
def deleteokFunc(request):
    delData=MyqnaQna.objects.get(id=request.POST.get('id'))
    if delData.passwd == request.POST.get('del_passwd'):
        delData.delete()
        return HttpResponseRedirect('/qna/list')
    else:
        return render(request,'error.html')
    pass
