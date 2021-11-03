"""Views file"""
import json
import os
import pickle
import time
import base64
import numpy as np
import matplotlib.pyplot as plt

from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from skimage.transform import resize

from .models import Images_and_Prediction
from .utils import detect_numbers, init, getimage


# Create your views here.

def index(request):
    """Returns index.html page to front end"""
    return render(request, 'main/index.html', {})


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@csrf_exempt
def predict(request):
    """
    Gets an image from front end does pre processing on it,
    resizes the image to 28x28 and runs it through neural net.
    :param request: Incoming image from front-end.
    :return: Http response prediction, confidence, fig_name to read.
    """
    model = init()

    """Getting image from front end"""
    getimage(request)

    """Detecting Numbers"""
    detected_numbers, cropped_numbers = detect_numbers()
    print(cropped_numbers)

    """Resizing to feed into the network"""
    to_predict = []
    for cropped in cropped_numbers:
        img = resize(cropped, (28, 28))
        img /= 255
        img = img.reshape(28, 28, 1)
        img = np.array(img)
        to_predict.append(img)

    """Convert into Numpy array"""
    to_predict = np.array(to_predict)

    """Getting rows and columns of plots"""
    columns = int(np.sqrt(len(to_predict)))
    rows = int(np.ceil(len(to_predict) / float(columns)))

    """Plotting images"""

    """Generate predictions for samples"""
    predictions = model.predict(to_predict)
    print(predictions)

    """Getting rows and columns of plots"""
    columns = int(np.sqrt(len(to_predict)))
    rows = int(np.ceil(len(to_predict) / float(columns)))

    """Plotting images"""
    preds = np.argmax(predictions, axis=1)
    fig = plt.figure(figsize=(12, 12))
    for i in range(1, len(to_predict) + 1):
        img = to_predict[i - 1].reshape(28, 28)
        fig_image = fig.add_subplot(rows, columns, i)
        fig_image.set_title('Prediction: ' + str(preds[i - 1]), fontsize=30)
        plt.imshow(img)

    directory = 'main/static/resources'
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))
    fig_name = str(time.time())
    plt.savefig('main/static/resources/' + fig_name + '.jpg')

    """Generate arg maxes for predictions"""
    response = np.argmax(predictions, axis=1).tolist()
    print(response)
    image = base64.b64encode(pickle.dumps(detected_numbers))
    detected_numbers_list = base64.b64encode(pickle.dumps(cropped_numbers))
    preds = response
    data = Images_and_Prediction(image=image, detected_numbers=detected_numbers_list, prediction=preds)
    data.save()
    json_object = {
        'output': response,
        'prediction': json.dumps(predictions, cls=NumpyEncoder),
        'fig_name': fig_name
    }
    return HttpResponse(json.dumps(json_object), content_type='application/json')
