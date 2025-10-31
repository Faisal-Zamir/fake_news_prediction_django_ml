from unittest import result
from aiohttp import request
from django.http import JsonResponse
from django.shortcuts import render
from news_classifier.Model_Files.fake_news_predictor import predict_fake_news
def homepage(request):
    if request.method == "POST" and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        news_text = request.POST.get("news_text", "")
        if news_text.strip():
            result = predict_fake_news(news_text)
            # Convert np.float64 to float
            for key in ['fake_probability', 'real_probability', 'confidence']:
                result[key] = float(result[key])
            return JsonResponse(result)
        else:
            return JsonResponse({"error": "Empty news text"})

    return render(request, 'news_classifier/homepage.html')

# Moving to Ajax for better UX 
# added : and request.headers.get('x-requested-with') == 'XMLHttpRequest':
# no need of result variable 
# return JsonResponse with converted float values to template