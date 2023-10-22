from os import path, makedirs
from datetime import datetime as dt
import json
from functools import partial
from io import BytesIO
from glob import glob
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from torch import nn
from flask import Flask, request, send_from_directory
from flask_cors import CORS

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

def get_mask_and_overlayed_images(palette, image, prediction):
    color_map = {i : k for i, k in enumerate(palette)}
    vis = np.zeros(prediction.shape + (3,))
    for i, c in color_map.items():
        vis[prediction == i] = color_map[i]
    mask = Image.fromarray(vis.astype(np.uint8))
    overlayed = Image.blend(image.convert("RGB"), mask.convert("RGB"), 0.5)
    return mask, overlayed

def visualize(labels, image, prediction):
    classes, palette = labels['classes'], labels['palette']
    mask, overlayed = get_mask_and_overlayed_images(palette, image, prediction)

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
    ax01.imshow(overlayed)
    ax0.bar(kinds, ratios, color=colors)
    return fig

# ---

makedirs('_analyzed/src_image', exist_ok=True)
makedirs('_analyzed/image', exist_ok=True)
makedirs('_analyzed/json', exist_ok=True)
makedirs('_analyzed/graph', exist_ok=True)

app = Flask(__name__)
CORS(app)

@app.route("/")
def root():
    return "Cesium Terrain Server Worked!"

cc_predict = partial(predict, *setup("nvidia/segformer-b5-finetuned-cityscapes-1024-1024"))
cc_labels = json.load(open('cityscapes_labels.json'))

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    f = BytesIO()
    file.save(f)
    image = Image.open(f).convert('RGB')
    prediction = cc_predict(image)
    mask, overlayed = get_mask_and_overlayed_images(cc_labels['palette'], image, prediction)
    hist, bins = np.histogram(prediction, range(len(cc_labels['classes'])))
    # hist = (hist * 100 / (image.size[0] * image.size[1])).astype(np.int32)
    
    name = dt.now().strftime("%Y%m%d_%H%M%S")
    image.save(f'_analyzed/src_image/{name}.png')
    overlayed.save(f'_analyzed/image/{name}.png')
    result = dict(request.form) | {
        'name': name,
        "width": image.width,
        "height": image.height,
        'histogram': hist.astype(np.int32).tolist()
    }
    json.dump(result, open(f'_analyzed/json/{name}.json', 'w'))
    graph = visualize(cc_labels, image, prediction)
    graph.savefig(f'_analyzed/graph/{name}.png')
    return result

@app.route('/analyzed/image/<path:filename>')
def get_analyzed_image(filename):
    return send_from_directory('_analyzed/image', filename)

@app.route('/analyzed/json/<path:filename>')
def get_analyzed_json(filename):
    return send_from_directory('_analyzed/json', filename)

@app.route('/analyzed/graph/<path:filename>')
def get_analyzed_graph(filename):
    return send_from_directory('_analyzed/graph', filename)

@app.route('/analyzed/whole-json')
def get_whole_json():
    whole_json = []
    for f in glob('_analyzed/json/*.json'):
        whole_json.append(json.load(open(f)))
    return whole_json

if __name__ == ('__main__'):
    app.run(debug=True, host='0.0.0.0')