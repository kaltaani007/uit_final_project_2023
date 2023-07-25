import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="Leaf Diseases Classification"
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.set_option('deprecation.showfileUploaderEncoding', False)
#@st.cache_data(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('tom-pot.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
         # Potato Disease Classification
         """
         )

file = st.file_uploader("", type=["jpg", "png" , "jfif"])
def import_and_predict(image_data, model):
        size = (256,256)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Pepper__bell___Bacterial_spot',
                
       'Pepper__bell___healthy',
     'Potato___Early_blight',
   'Potato___Late_blight',
 'Potato___healthy',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']
    
    string = "Prediction : " + class_names[np.argmax(predictions)]
    string2 = "Confidence rate " + str(np.max(predictions[0]))

  
   st.success(string)
   st.success(string2)

