from django.urls import path
from . import views


urlpatterns = [
    path('', views.app_view, name='App'),
    path('SmartPath/', views.app_view, name='App'),
    path('LogIn/', views.login_view, name='LogIn'),
    path('SignUp/', views.sign_view, name='SignUp'),
    path('Home/', views.homepage, name='Home'),
    path('MyAcademicPath/', views.academic_path, name='MyAcademicPath'),
    path('logout/', views.logout_view, name='LogOut'),
    path('save-goal/', views.save_goal, name='save_goal'),
    path('save-department/', views.save_department, name='save_department'),  
    path('curriculum-map/<int:department_id>/', views.curriculum_map, name='curriculum_map'),
    path('Profile/', views.profile_view, name='Profile'),
    path('save-plan/', views.save_plan, name='save_plan'),
    path('delete-plan/', views.delete_plan, name='delete_plan'),
 
]