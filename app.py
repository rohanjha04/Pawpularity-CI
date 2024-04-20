import streamlit as st
from PIL import Image
from load_model import predict_image, load_model, load_image, get_metadata


def main():
    st.title('Image Pawpularity Estimator')
    metadata_columns = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']
    model= load_model('pawpularity_model.weights.h5')
    uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        attribute_list = [0] * 12
        for i in range(12):
            attribute_list[i] = st.checkbox(f'{metadata_columns[i]}', value=False)

        # Button to trigger image processing
        if st.button('Process Image'):
            #Make image compatible for the model
            image = load_image(image)
            #Convert True/False to 1/0
            metadata = get_metadata(attribute_list)
            pawpularity_value = predict_image(image, metadata, model)
            # st.write('pawpularity Value:', pawpularity_value[0][0])
            st.markdown(f'<p style="font-size:40px">Pawpularity Value: {pawpularity_value[0][0]}</p>', unsafe_allow_html=True)
        
if __name__ == '__main__':
    main()
