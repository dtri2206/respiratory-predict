import streamlit as st
import numpy as np
import librosa as lb
import keras
import pywt

# Load your pre-trained model
model = keras.models.load_model("my_model.h5")  # Adjust based on your model format

def wavelet_transform(signal, wavelet='db1', level=5, target_length=259):
    """
    Apply wavelet transform and reshape the output to be compatible with CNN input.
    Returns a 2D array of shape (20, target_length)
    """
    # Pad or truncate the signal to ensure consistent length
    if len(signal) > 132300:
        signal = signal[:132300]
    else:
        signal = np.pad(signal, (0, 132300 - len(signal)), 'constant')
    
    # Apply wavelet transform
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Concatenate all coefficients
    all_coeffs = np.concatenate([coeff for coeff in coeffs])
    
    # Calculate number of segments needed
    total_segments = 20
    
    # Reshape the coefficients into segments
    segment_length = len(all_coeffs) // total_segments
    segments = []
    
    for i in range(total_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment = all_coeffs[start_idx:end_idx]
        
        # Resample segment to target length using linear interpolation
        x_original = np.linspace(0, 1, len(segment))
        x_target = np.linspace(0, 1, target_length)
        segment_resampled = np.interp(x_target, x_original, segment)
        
        segments.append(segment_resampled)
    
    # Stack segments into final shape (20, 259)
    result = np.stack(segments)
    
    return result

def get_features(path, target_length=259):
    # load audio file
    soundArr, sample_rate = lb.load(path)
    
    # trích xuất đặc trưng MFCC 
    mfcc = lb.feature.mfcc(y=soundArr, sr=sample_rate, n_mfcc=20)
    
    # đảm bảo MFCC shape through interpolation
    if mfcc.shape[1] != target_length:
        mfcc_resized = np.zeros((20, target_length))
        for i in range(20):
            x_orig = np.linspace(0, 1, mfcc.shape[1])
            x_new = np.linspace(0, 1, target_length)
            mfcc_resized[i] = np.interp(x_new, x_orig, mfcc[i])
        mfcc = mfcc_resized
    wavelet = wavelet_transform(soundArr, wavelet='db1', level=5, target_length=target_length)
    mfcc = np.expand_dims(mfcc, axis=-1)      # Shape: (20, 259, 1)
    wavelet = np.expand_dims(wavelet, axis=-1) # Shape: (20, 259, 1)

    return mfcc, wavelet
    
    # dùng wavelet transform
    wavelet = wavelet_transform(soundArr, wavelet='db1', level=5, target_length=target_length)
    
    
    # thêm channel dimension cho CNN
    mfcc = np.expand_dims(mfcc, axis=-1)      # Shape: (20, 259, 1)
    wavelet = np.expand_dims(wavelet, axis=-1) # Shape: (20, 259, 1)
    
    return mfcc, wavelet
# Streamlit application
st.title("Respiratory Sound Classification")

# File uploader
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    mfcc, wavelet= [], []
    path = uploaded_file
    a, b = get_features(path)
    mfcc.append(a)
    wavelet.append(b)
    mfcc = np.array(mfcc)
    wavelet = np.array(wavelet)
    # Make a prediction
    prediction = model.predict({"mfcc": mfcc, "wavelet": wavelet})  # Adjust based on your model input
    predicted_labels = np.argmax(prediction, axis=1)
    if predicted_labels == 1:
            predicted_labels = 'COPD'
    elif predicted_labels == 0:
            predicted_labels = 'asthma'
    elif predicted_labels == 2:
            predicted_labels = 'HEALTHY'
    else:
            predicted_labels = 'pneumonia'
    # Display the prediction result
    st.markdown(f"<h2 style='text-align: center; color: red;'>Predicted Class: {predicted_labels}</h2>", unsafe_allow_html=True)
    # Button to upload another file
    if st.button("Upload Another File"):
        st.experimental_rerun()  # Reload the app to allow for a new upload
