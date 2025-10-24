from django.shortcuts import render

def homepage(request):
    return render(request, 'news_classifier/homepage.html')