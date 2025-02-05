import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model= tf.keras.models.load_model("trained_plant_disease_model.keras")
    image= tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions= model.predict(input_arr)
    return np.argmax(predictions)
st.sidebar.title("Potato Leaf Disease Detection")
app_mode = st.sidebar.selectbox('select page',['Home','Disease Recognition'])


if(app_mode=='HOME'):
    st.markdown("<h1 style='text-align: center;'>Potato Leaf Disease Detection", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>A Plant Disease System for Sustainable Agriculture focuses on early detection, prevention, and management of plant diseases to ensure the health of crops while minimizing the environmental impact of agricultural practices.It promotes sustainable farming practices such as precision agriculture, integrated pest management, and biocontrol.The system helps reduce chemical use, increase crop yields, and protect the environment. It also provides farmers with real-time insights, disease forecasting, and personalized recommendations.", unsafe_allow_html=True)

elif(app_mode=='Disease Recognition'):
    st.header('Potato Leaf Disease Detection')
    st.markdown("<p style='text-align: center;'>A Plant Disease System for Sustainable Agriculture focuses on early detection, prevention, and management of plant diseases to ensure the health of crops while minimizing the environmental impact of agricultural practices.It promotes sustainable farming practices such as precision agriculture, integrated pest management, and biocontrol.The system helps reduce chemical use, increase crop yields, and protect the environment. It also provides farmers with real-time insights, disease forecasting, and personalized recommendations.", unsafe_allow_html=True)


test_image= st.file_uploader('Choose an image:')
if(st.button('Show Image')):
    st.image(test_image,width=4,use_column_width=True)

if (st.button('Predict')):
    st.snow()
    st.write('our prediction')
    result_index = model_prediction(test_image)
    class_name=['Potato___Early_blight','Potato___Late_blight','Potato___healthy']
    st.success('Model is predicting its a {}'.format(class_name[result_index]))