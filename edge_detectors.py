import streamlit as st 
import numpy as np
from cv2 import cv2 as cv
from PIL import Image
st.set_page_config(page_icon="./favicon.png", page_title="Edge Detector - Ziaul Karim")
st.title ("Edge Detector")
st.markdown("Detect Edges from Images, by using different techniques like 👉 Canny, Sobel, Laplacian & Prewitt Edge Detectors.")
st.markdown("An example Image is given below 👇")
file= st.file_uploader("Upload Image")

@st.cache()

def load_file(image):
    if image is not None:
        image = Image.open(image)
        return image
    else:
        image = Image.open('rose.jpg')
        return image
img= load_file(file)
img=np.array(img)
def canny(img, t1,t2):
    img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (3,3), 0)
    edges = cv.Canny(img,t1,t2)
    return edges
edge_detector= st.selectbox("Select an Edge Detector", ["Click here to select...","Canny", "Sobel", "Laplacian", "Prewitt"])

def sobel(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (3,3), 0)
    sobelx = cv.Sobel(src=img_blur, ddepth=cv.CV_8U, dx=1, dy=0, ksize=5)
    sobely = cv.Sobel(src=img_blur, ddepth=cv.CV_8U, dx=0, dy=1, ksize=5)
    sobelxy = cv.Sobel(src=img_blur, ddepth=cv.CV_8U, dx=1, dy=1, ksize=5)
    return sobelx, sobely,sobelxy
def prewitt(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gaussian = cv.GaussianBlur(gray,(3,3),0)
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv.filter2D(img_gaussian, -1, kernely)
    return img_prewittx, img_prewitty
def laplacian(img):
    img=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    laplacian = cv.Laplacian(img,cv.CV_8U)
    return laplacian
if edge_detector == "Canny":
    st.subheader("Select Threshold values")
    min_threshold=st.slider(min_value=0, max_value=200,label='Minimum Threshold')
    max_threshold=st.slider(min_value=0, max_value= 200, label='Maximum Threshold')
    st.image([img,canny(img, min_threshold, max_threshold)], caption=['Original', 'Canny'], width=200)
elif edge_detector == "Sobel":
    col1, col2 = st.columns(2)
    with col1:
        st.header("Original")
        st.image(img, caption="Original", width=200)
    with col2:
        st.header("Sobel")
        st.image([i for i in sobel(img)], caption=["Sobel X", "Sobel Y", "Sobel XY"], width=150)
elif edge_detector == "Prewitt":
    col1, col2 = st.columns(2)
    with col1:
        st.header("Original")
        st.image(img, caption="Original", width=200)
    with col2:
        st.header("Prewitt")
        st.image([i for i in prewitt(img)], caption=["Prewitt X", "Prewitt Y"], width=180)
elif edge_detector == "Laplacian":
    st.image([img,laplacian(img)],caption=["Original", "Laplacian"],width=200)
else:
    st.markdown("You haven't selected anything from the dropdown box yet.")


hide_streamlit_style = """
            <head>
            <link rel="shortcut icon" href="./favicon.png">
            <style>
            #MainMenu{visibility: hidden;}
            .css-fk4es0{display:none;}
            .css-1lsmgbg {display: none;}
            .myFooter{color:#fff; margin-top: 150px; text-align: center;}
            .myFooter a{color: #DB8962; font-weight: bolder;}
            .css-10trblm{color:#DB8962; text-align:center;}
            .css-16huue1 {color:#DB8962; font-size:18px;}
            .css-v37k9u p{color:#edf5e1; font-size: 18px;}
            .css-1q8dd3e{color:#DB8962;}
            .css-1q8dd3e:hover{color:#edf5e1; border-color:#DB8962;}
            </style>
            <title> Edge Detectors </title>
            </head>
            <div class="myFooter">© 2022 Copyright | Made by <a href="https://codingwithzk.netlify.app" >Md. Ziaul Karim</a> <br> with <a href="https://streamlit.io/"> Streamlit </a> </div>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 