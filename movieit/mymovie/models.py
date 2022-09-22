from django.db import models

# Create your models here.

class NoticeTab(models.Model):
    name = models.CharField(max_length = 20) # 이름
    passwd = models.CharField(max_length = 20) # 비밀번호
    title = models.CharField(max_length = 100) # 제목
    cont = models.TextField() # 내용
    nip = models.GenericIPAddressField() # ip
    ndate = models.DateTimeField() # 등록일
    readcnt = models.IntegerField() # 조회수
    likecnt = models.IntegerField() # 좋아요수