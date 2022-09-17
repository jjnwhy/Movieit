from datetime import datetime
from django.http.response import HttpResponseRedirect
from django.shortcuts import render
from myqna.models import MyqnaQna


def replyFunc(request):
    try:
        data=MyqnaQna.objects.get(id=request.GET.get('id'))
        context={'data_one':data}
        return render(request,'rep/reply.html',context)
    except Exception as e:
        print('댓글 대상 원글 읽기 오류: ',e)
        return render(request,'error.html')
def replyokFunc(request):
    if request.method=='POST':
        try:
            #print(request.POST.get('id'),request.POST.get('name'))
            # onum처리
            repGnum=int(request.POST.get('gnum'))
            repOnum=int(request.POST.get('onum'))
            imsiRec=MyqnaQna.objects.get(id=request.POST.get('id'))
            oldGnum=imsiRec.gnum
            oldOnum=imsiRec.onum
            # onum갱신
            if oldOnum >= repOnum and oldGnum == repGnum:
                oldOnum=oldOnum+1
            #댓글 저장
            MyqnaQna(
                name=request.POST.get('name'),
                passwd=request.POST.get('passwd'),
                mail=request.POST.get('mail'),
                title=request.POST.get('title'),
                cont=request.POST.get('cont'),
                bip=request.META['REMOTE_ADDR'], #request.META.remote('REMOTE_ADDR')
                bdate=datetime.now(),
                readcnt=0,
                gnum=repGnum,
                onum=oldOnum,
                nested=int(request.POST.get('nested'))+1,                
            ).save()
            return HttpResponseRedirect('/qna/list') #댓글 작성 후 목록 보기.
        except Exception as e:
            print('댓글 저장 오류: ',e)
            return render(request,'error.html')