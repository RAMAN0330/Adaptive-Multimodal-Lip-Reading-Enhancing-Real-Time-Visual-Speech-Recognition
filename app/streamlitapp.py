# import necessary dependencies
import streamlit as st
import os
import imageio
import numpy as np
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
import subprocess

# set the layout of the Streamlit app to wide
st.set_page_config(layout='wide')

# set up the sidebar
with st.sidebar:
    # display logo and app title
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')

# display app title
st.title('Lip Reading app')

# generate a list of options for videos
options = os.listdir(os.path.join('..','data','s1'))
selected_video = st.selectbox('Choose Video',options)

# set up two columns for displaying the video and model output
col1, col2 = st.columns(2)

if options:
    # render the selected video
    with col1:
        st.info('The video below displays the selected video in mpg format')
        data_dir = "/Users/raman/Desktop/Deep Learning/lipread_tensorflow-main/data/s1"
        # set up file paths for input video
        file_path = os.path.join(data_dir, selected_video)

        if os.path.exists(file_path):
            # Check if the video file is of a valid format
            if file_path.endswith(('.mpg', '.mpeg')):
                with open(file_path, 'rb') as video:
                    video_bytes = video.read()
                    st.video(video_bytes, format="video/mpg")  # You can also try with format="video/mpeg"
            else:
                st.error('Unsupported video format. Please upload a .mpg or .mpeg file.')
        else:
            # display an error message if video doesn't exist
            st.error(f'Error: Video file not found: {file_path}')
    # display the video as input to the model
    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        # load the video and its annotations
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        # convert the video to a GIF for display
        st.image('Animation.gif',width=400)

        # display the output of the model as tokens
        st.info('This is the output of the machine learning as tokens')
        model = load_model()
        # predict the text from the video
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # convert the model output from tokens to text
        st.info('This is the decoded text')
        converted_preds = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_preds)
