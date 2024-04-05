import io
import torch
from flask import Flask, render_template, request, jsonify, Response
from PIL import Image, ImageFilter
import torchvision.transforms as transforms

#allow request from different ports
from flask import Flask, render_template
from flask_cors import CORS

from model import SuperResolutionAutoencoder


model = SuperResolutionAutoencoder()
model.load_state_dict(torch.load('weights/super_resolution_autoencoder_epoch_6.pth'))
model.eval()

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data", methods=["POST"])
def data():
    im = Image.open(io.BytesIO(request.data)).convert("RGB")
    blurred_image = im.filter(ImageFilter.GaussianBlur(2))

    transform = transforms.ToTensor()     
    input_tensor = transform(blurred_image).unsqueeze(0)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))

    img_byte_array = io.BytesIO()
    output_image.save(img_byte_array, format='JPEG', quality=100)
    img_byte_array.seek(0)

    # Return the processed image as a response
    return Response(img_byte_array, mimetype='image/jpeg')
    
    
