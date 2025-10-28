from unittest import result
from django.shortcuts import render
from news_classifier.Model_Files.fake_news_predictor import predict_fake_news
def homepage(request):
    result = None
    if request.method == "POST":
        news_text = request.POST.get("news_text", "")
        if news_text.strip():
            result = predict_fake_news(news_text)
            # Convert np.float64 to float
            for key in ['fake_probability', 'real_probability', 'confidence']:
                result[key] = float(result[key])


    print("Result:", result)
    
    context = {
        "result": result
    }
    return render(request, 'news_classifier/homepage.html',context)