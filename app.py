from os import path
from datetime import datetime as dt
import json
from functools import partial
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from torch import nn
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

matplotlib.use('Agg')

def get_body(fpath):
    return path.splitext(path.basename(fpath))[0]

def setup(model_name):
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    return feature_extractor, model

def predict(feature_extractor, model, image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    upsampled_logits = nn.functional.interpolate(logits,
                    size=(image.size[1], image.size[0]), # (height, width)
                    mode='bilinear',
                    align_corners=False)
        
    predicted_mask = upsampled_logits.argmax(dim=1).cpu().numpy()
    return predicted_mask[0]

plt.ioff()

def visualize(labels, image, prediction):
    classes, palette = labels['classes'], labels['palette']
    color_map = {i : k for i, k in enumerate(palette)}
    vis = np.zeros(prediction.shape + (3,))
    for i, c in color_map.items():
        vis[prediction == i] = color_map[i]
    mask = Image.fromarray(vis.astype(np.uint8))
    overlayed_img = Image.blend(image.convert("RGB"), mask.convert("RGB"), 0.5)

    hist, bins = np.histogram(prediction, range(0, len(classes)))
    histbins = sorted([(h, b)for h, b in zip(hist, bins)], reverse=True)
    n_o_pixels = mask.size[0] * mask.size[1]
    stats = [(classes[b], (np.array(palette[b]) / 255).tolist(), r) for h, b in histbins if (r:=(int(h) * 100 / n_o_pixels)) > 0.1]
    kinds, colors, ratios = list(zip(*stats))

    fig = plt.figure(figsize=(20, 10), layout="constrained")
    spec = fig.add_gridspec(2, 2)
    ax00 = fig.add_subplot(spec[0, 0])
    ax01 = fig.add_subplot(spec[0, 1])
    ax0 = fig.add_subplot(spec[1, :])
    ax00.imshow(mask)
    ax01.imshow(overlayed_img)
    ax0.bar(kinds, ratios, color=colors)
    return fig

# ---

app = Flask(__name__)
CORS(app)

@app.route("/")
def root():
    return "Cesium Terrain Server Worked!"

cc_predict = partial(predict, *setup("nvidia/segformer-b5-finetuned-cityscapes-1024-1024"))
cc_labels = json.load(open('cityscapes_labels.json'))
cc_visualize = partial(visualize, cc_labels)

# @app.route('/analyse/cityscape/<path:path>')
def cc_analyze(url):
    image = Image.open(url).convert('RGB')
    prediction = cc_predict(image)
    hist, bins = np.histogram(prediction, range(len(cc_labels['classes'])))
    hist = (hist * 100 / (image.size[0] * image.size[1])).astype(np.int32)
    fig = cc_visualize(image, prediction)
    # body = get_body(url)
    # json.dump(hist.tolist(), open(f'_segmented/{body}.json', 'w'))
    # fig.savefig(f'_uploaded/{body}.png')
    basename = f'{dt.now().strftime("%Y%m%d_%H%M%S")}.png'
    fig.savefig(f'_tmp/{basename}')
    return {
        'histogram': hist.tolist(),
        'graph': basename
    }

from io import BytesIO

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    f = BytesIO()
    file.save(f)
    result = cc_analyze(f)
    return f'{json.dumps(result)}'

@app.route('/graph/<path:filename>')
def static_file(filename):
    print(filename)
    return send_from_directory('_tmp', filename)

# app.run(debug=True)
# flask run --debug