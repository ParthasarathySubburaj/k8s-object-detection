import numpy as np
from PIL import Image
import streamlit as st
from detect import SSDModel

st.beta_set_page_config(
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title="K8S deployment",  # String or None. Strings get appended with "• Streamlit".
    page_icon=None,  # String, anything supported by st.image, or None.
)

readme_text = st.markdown(open("README.md").read())


@st.cache
def load_model():
    model = SSDModel('checkpoint_ssd300.pth.tar')
    print("Loading model...")
    return model


def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["None", "mode1", "mode2", "mode3"])
    if app_mode == "mode1":
        run_model_1()
        pass
    elif app_mode == "mode2":
        # run_model_2()
        pass
    else:
        # run_model_3()
        pass


def run_model_1():
    readme_text.empty()
    st.markdown("SSD DETECTOR")
    model = load_model()
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer, mode='r')
        image = image.convert('RGB')
        original_image = np.copy(image)
        _, labels = model.detect(image, min_score=0.2, max_overlap=0.5)
        st.image([original_image, image], caption=["Original Image", "Inferred Image"], width=450)
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
