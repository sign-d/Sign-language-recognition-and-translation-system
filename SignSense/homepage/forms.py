from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import Sign
from django.contrib.auth import authenticate

class SignUpForm(UserCreationForm):
    class Meta:
        model = Sign
        fields = ('email', 'password1', 'password2')

    def clean_password2(self):
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Passwords don't match")
        if len(password1) < 6:
            raise forms.ValidationError("Password must be at least 6 characters long")
        return password2

class LoginForm(forms.Form):
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)

    def clean(self):
        email = self.cleaned_data.get('email')
        password = self.cleaned_data.get('password')

        if email and password:
            user = authenticate(email=email, password=password)
            if user is None:
                raise forms.ValidationError("Invalid login credentials")
        return super().clean()