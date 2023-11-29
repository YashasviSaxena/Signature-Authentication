
import streamlit as st
# from tensorflow.keras.models import load_model

st.set_option('deprication.showfileUploadingEncoding',False)
@st.cache(allow_output_mutation = True)
def load_model1():
    model =tf.keras.models.load_model('Signature Authentication/my_model2.hdf5')
    return model
model = load_model()
st.write("""
            #Signature Authentication
            """
        )

file = sr.file_uploader("Please upload an signature image" , type = ["jpg","png"])
import cv2
from PIL import Image,Image Ops
import numpy as np
def import_and_predict(image_data,model):
    size = (128,128)
    image = ImageOps.fit(image_data,Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predit(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else :
    image = Image.open(file)
    st.image(image,use_column_width =True)
    predictions = import_and_predict(image,model)
    if predictions < 0.5:
        string = "The signature is real."
    else:
        string = "The signature is forged."
    st.success(string)
    
