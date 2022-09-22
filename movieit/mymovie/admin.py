from django.contrib import admin
from mymovie.models import NoticeTab

# Register your models here.
class MyMovieAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'passwd', 'title', 'cont', 'nip', 'ndate', 'readcnt', 'likecnt')

admin.site.register(NoticeTab, MyMovieAdmin)