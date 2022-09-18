from django.shortcuts import render, redirect
from mymovie.models import NoticeTab
from datetime import datetime
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.http.response import HttpResponseRedirect


def mainFunc(request):
    data_all = NoticeTab.objects.all().order_by('-id')
    per_page = 10
    paginator = Paginator(data_all, per_page)
    page = request.GET.get('page')
    try:
        datas = paginator.page(page)
    except PageNotAnInteger:
        datas = paginator.page(1)
    except EmptyPage:
        datas = paginator.page(paginator.num_pages)
        
    return render(request, 'notice.html', {'datas': datas})

def insertFunc(request):
    return render(request, 'insert.html')

def insertokFunc(request):
    if request.method == "POST":
        try:
            datas = NoticeTab.objects.all()
            NoticeTab(
                name = request.POST.get('name'),
                passwd = request.POST.get('passwd'),
                title = request.POST.get('title'),
                cont = request.POST.get('cont'),
                nip = request.META['REMOTE_ADDR'],
                ndate = datetime.now(),
                readcnt = 0,
                likecnt = 0
            ).save()
             
        except Exception as e:
            print('공지사항 추가 error:', e)
            return render(request, 'error.html')
    return HttpResponseRedirect('/notice') # 게시글 추가 후 메인으로 돌아가기

def searchFunc(request):
    if request.method == "POST":
        s_type = request.POST.get("s_type")
        s_value = request.POST.get('s_value')
        
        if s_type == "title":
            datas_search = NoticeTab.objects.filter(title__contains=s_value).order_by('-id')
        elif s_type == "name":
            datas_search = NoticeTab.objects.filter(name__contains=s_value).order_by('-id')
        
        per_page = 10
        paginator = Paginator(datas_search, per_page)
        page = request.GET.get('page')
        try:
            datas = paginator.page(page)
        except PageNotAnInteger:
            datas = paginator.page(1)
        except EmptyPage:
            datas = paginator.page(paginator.num_pages)
    return render(request, 'notice.html', {'datas': datas})

def contentFunc(request):
    page = request.GET.get('page')
    data = NoticeTab.objects.get(id=request.GET.get('id'))
    data.readcnt = data.readcnt + 1
    data.save()
    
    return render(request, 'content.html', {'data_one':data, 'page':page})

def contentokFunc(request):
    # print('data')
    data = NoticeTab.objects.get(id=request.GET.get('id'))
    data.likecnt = data.likecnt + 1
    data.save()
    return render(request, 'content.html', {'data_one':data})
    # return redirect('/notice/content')

def updateFunc(request):
    try:
        upData = NoticeTab.objects.get(id=request.GET.get('id'))
    except Exception as e:
        print('수정자료 읽기 오류:', e)
        return render(request, 'error.html')
    return render(request, 'update.html', {'data':upData})

def updateokFunc(request):
    try:
        upData = NoticeTab.objects.get(id=request.POST.get('id'))
        
        if upData.passwd == request.POST.get('up_passwd'):
            upData.name = request.POST.get('name')
            upData.title = request.POST.get('title')
            upData.cont = request.POST.get('cont')
            upData.save()
        else:
            return render(request, 'update.html', {'data':upData})
    except Exception as e:
        print("수정자료 읽기 오류:", e)
        return render(request, 'error.html')
    return redirect('/notice')

def deleteFunc(request):
    try:
        delData = NoticeTab.objects.get(id=request.GET.get('id'))
    except Exception as e:
        print('삭제자료 읽기 오류:', e)
        return render(request, 'error.html')
    return render(request, 'delete.html', {'data':delData})

def deleteokFunc(request):
    delData = NoticeTab.objects.get(id=request.POST.get('id'))
    
    if delData.passwd == request.POST.get('del_passwd'):
        delData.delete()
        return redirect('/notice')
    else:
        return render(request, 'error.html') 
    
    
def get_client_ip(request): # 수정중
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


    