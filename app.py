import gradio as gr
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# Adjusted color map with Black settings
color_map = {
    "Green": 60,
    "Blue": 120,
    "Yellow": 30,
    "Purple": 150,
    "Black": 0 
}

def apply_filter(image, selected_color):
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu()

    upsampled_logits = F.interpolate(
        logits,
        size=image.size[::-1], 
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    hair_mask = (pred_seg == 2).astype(np.uint8)

    image_cv = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    hsv_image = cv.cvtColor(image_cv, cv.COLOR_BGR2HSV)

    hue_value = color_map.get(selected_color, 0)

    if selected_color == "Black":
        hsv_image[..., 1] = np.where(hair_mask == 1, 0, hsv_image[..., 1])   
        hsv_image[..., 2] = np.where(hair_mask == 1, 30, hsv_image[..., 2]) 
    else:
        hsv_image[..., 0] = np.where(hair_mask == 1, hue_value, hsv_image[..., 0])
        hsv_image[..., 1] = np.where(hair_mask == 1, hsv_image[..., 1] * 1.5, hsv_image[..., 1])

    final_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

    return Image.fromarray(cv.cvtColor(final_image, cv.COLOR_BGR2RGB))

iface = gr.Interface(
    fn=apply_filter, 
    inputs=[
        gr.Image(type="pil"), 
        gr.Dropdown(
            choices=["Green", "Blue", "Yellow", "Purple", "Black"], 
            label="Select Color"
        )
    ], 
    outputs=gr.Image(type="numpy", label="result"),
    live=False, 
    title="Saç Rengi Değiştirme",
    description="Saçınızın rengini değiştirmek için SUBMIT butonuna tıklayınız."
)

iface.launch(share=True)
