import io
import torch
from flask import Flask, render_template, request, jsonify, Response
from PIL import Image, ImageFilter
import torchvision.transforms as transforms

#allow request from different ports
from flask import Flask, render_template
from flask_cors import CORS

from model import SuperResolutionAutoencoder

def preprocess_image(image):
    """
    transform = transforms.Compose([
        #transforms.Resize((178, 218)),  # Resize to match model input size
        transforms.ToTensor()          # Convert to tensor
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    #image = transform(image).unsqueeze(0)  # Add batch dimension
    image = transform(image)
    """

    transform = transforms.ToTensor()     
    image = transform(image).unsqueeze(0)
    return image


model = SuperResolutionAutoencoder()
model.load_state_dict(torch.load('overfit_weights/super_resolution_autoencoder_epoch_6.pth'))
#model.load_state_dict(torch.load('weights10.pth'))

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data", methods=["POST"])
def data():
    im = Image.open(io.BytesIO(request.data)).convert("RGB")
    blurred_image = im.filter(ImageFilter.GaussianBlur(radius=2))
    #blurred_image.show()

    input_image = preprocess_image(blurred_image)
    #input_image = preprocess_image(im)

    output_image = model(input_image)

    # Display or save the output image
    output_image = output_image.squeeze(0)  # Remove batch dimension
    output_image = transforms.ToPILImage()(output_image)

    #size from CELEB A dataset
    #width, height = 178, 218
    #im = im.resize((width, height))
    img_byte_array = io.BytesIO()
    output_image.save(img_byte_array, format='JPEG')
    img_byte_array.seek(0)


    # Convert the processed image back to bytes
    """
    img_byte_array = io.BytesIO()
    im.save(img_byte_array, format='JPEG')
    img_byte_array.seek(0)
    """
    # Return the processed image as a response
    return Response(img_byte_array, mimetype='image/jpeg')
    
    
