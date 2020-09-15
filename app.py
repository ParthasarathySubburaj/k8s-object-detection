import numpy as np
from PIL import Image
import streamlit as st
from detect import SSDModel, YOLOV3

st.beta_set_page_config(
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title="K8S deployment",  # String or None. Strings get appended with "• Streamlit".
    page_icon=None,  # String, anything supported by st.image, or None.
)

readme_text = st.markdown(open("README.md").read())


@st.cache
def load_ssd_model():
    ssd_model = SSDModel('checkpoint_ssd300.pth.tar')
    print("Loading SSD model...")
    return ssd_model

@st.cache
def load_yolov3_model():
    yolov3_model = YOLOV3("config/yolov3.cfg", "weights/yolov3.weights", "data/coco.names")
    print("Loading YOLOV3 model...")
    return yolov3_model

def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["None", "SSD", "YOLOV3"])
    if app_mode == "SSD":
        run_model_ssd()
    elif app_mode == "YOLOV3":
        run_model_yolov3()


def run_model_ssd():
    readme_text.empty()
    st.markdown("SSD DETECTOR")
    ssd_model = load_ssd_model()
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer, mode='r')
        image = image.convert('RGB')
        original_image = np.copy(image)
        _, labels = ssd_model.detect(image, min_score=0.2, max_overlap=0.5)
        st.image([original_image, image], caption=["Original Image", "Inferred Image"], width=450)
        st.markdown("Objects detected")
        object_count = {}
        for i in labels:
            if i not in object_count:
                object_count[i] = 0
            object_count[i] = object_count[i] + 1
        for object, count in object_count.items():
            st.markdown("{} ----> {}".format(object, count))


def run_model_yolov3():
    readme_text.empty()
    st.markdown("YOLO V3 DETECTOR")
    yolov3_model = load_yolov3_model()
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer, mode='r')
        image = image.convert('RGB').resize((416, 416))
        original_image = np.copy(image)
        annotated_image, labels = yolov3_model.detect(image)
        st.image([original_image, annotated_image], caption=["Original Image", "Inferred Image"], width=450)
        st.markdown("Objects detected")
        object_count = {}
        for i in labels:
            if i not in object_count:
                object_count[i] = 0
            object_count[i] = object_count[i] + 1
        for object, count in object_count.items():
            st.markdown("{} ----> {}".format(object, count))


if __name__ == "__main__":
    main()
