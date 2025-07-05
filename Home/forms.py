from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import Student

class LoginForm(forms.Form):
    student_id = forms.CharField(
        label='رقم القيد',
        max_length=20,
        widget=forms.TextInput(attrs={
            'placeholder': 'أدخل رقم القيد',
            'class': 'form-control'
        })
    )
    password = forms.CharField(
        label='كلمة السر',
        widget=forms.PasswordInput(attrs={
            'placeholder': 'أدخل كلمة السر',
            'class': 'form-control'
        })
    )


class StudentRegistrationForm(UserCreationForm):
    password1 = forms.CharField(
        label="كلمة السر",
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'أدخل كلمة السر'})
    )
    password2 = forms.CharField(
        label="تأكيد كلمة السر",
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'أكد كلمة السر'})
    )

    class Meta:
        model = Student
        fields = ('student_id', 'email')
        widgets = {
            'student_id': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'أدخل رقم القيد'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'أدخل البريد الإلكتروني'}),
        }
        labels = {
            'student_id': 'رقم القيد',
            'email': 'البريد الإلكتروني',
        }