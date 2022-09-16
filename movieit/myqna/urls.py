from django.urls import path
from views import view1,view2

urlpatterns = [
    path('list',view1.listFunc),
    
    path('insert',view1.insertFunc),
    path('insertok',view1.insertokFunc),     
    
    path('search',view1.searchFunc),
    
    path('content',view1.contentFunc),
    
    path('update',view1.updateFunc),
    path('updateok',view1.updateokFunc),
     
    path('delete',view1.deleteFunc),
    path('deleteok',view1.deleteokFunc),
    
    # 댓글
    path('reply',view2.replyFunc), 
    path('replyok',view2.replyokFunc),
]
