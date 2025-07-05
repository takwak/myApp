from django.urls import path
from . import views

urlpatterns = [
    path('', views.app_view, name='home'),  
    path('SmartPath/', views.app_view, name='app'),  
    path('login/', views.login_view, name='login'),  
    path('signup/', views.sign_view, name='signup'),  
    path('home/', views.homepage, name='dashboard'),  
    path('my-academic-path/', views.academic_path, name='academic_path'),  
    path('logout/', views.logout_view, name='logout'),  
    path('save-goal/', views.save_goal, name='save_goal'),
    path('save-department/', views.save_department, name='save_department'),
    path('curriculum-map/<int:department_id>/', views.curriculum_map, name='curriculum_map'),
    path('profile/', views.profile_view, name='profile'),  
    path('save-plan/', views.save_plan, name='save_plan'),
    path('delete-plan/', views.delete_plan, name='delete_plan'),
]