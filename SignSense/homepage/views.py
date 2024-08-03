from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import AuthenticationForm
from .forms import SignUpForm, LoginForm
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage

import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load the model
model = tf.keras.models.load_model('C:\\Users\\Administrator\\Documents\\ActionSense\\SignSense\\homepage\\templates\\homepage\\model_alpha_1.h5')

def index(request):
    return render(request, 'homepage/index.html')

@csrf_exempt
def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            user = authenticate(request, email=email, password=password)
            if user is not None:
                login(request, user)
                return JsonResponse({'status': 'success'})
            else:
                return JsonResponse({'status': 'error', 'errors': ['Invalid login credentials']})
        else:
            errors = [error for error in form.errors.values()]
            return JsonResponse({'status': 'error', 'errors': errors})
    else:
        form = LoginForm()
    return render(request, 'homepage/login.html', {'form': form})

@csrf_exempt
def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            return JsonResponse({'status': 'success'})
        else:
            errors = [error for error in form.errors.values()]
            return JsonResponse({'status': 'error', 'errors': errors})
    else:
        form = SignUpForm()
    return render(request, 'homepage/signup.html', {'form': form})

def recog_view(request):
    return render(request, 'homepage/recog.html')

def about_view(request):
    return render(request, 'homepage/about.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if file:
            # Process the uploaded file
            image = Image.open(file)
            image = image.resize((224, 224))  # Resize if required by the model
            image_array = np.array(image) / 255.0  # Normalize if required
            image_array = np.expand_dims(image_array, axis=0)

            # Make a prediction
            predictions = model.predict(image_array)
            # Process predictions to desired format
            response = {
                'predictions': [
                    {'label': 'Sign A', 'confidence': float(predictions[0][0])},
                    {'label': 'Sign B', 'confidence': float(predictions[0][1])}
                ]
            }
            return JsonResponse(response)

    return JsonResponse({'error': 'Invalid request'}, status=400)