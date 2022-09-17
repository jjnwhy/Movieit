from django.contrib import admin

from myqna.models import MyqnaQna
# Register your models here.
class MyQnaAdmin(admin.ModelAdmin):
    list_display=('id','name','passwd','mail','title','cont','bip','bdate','readcnt','gnum','onum','nested')

admin.site.register(MyqnaQna,MyQnaAdmin)