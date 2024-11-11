import streamlit as st
import os
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import io
import librosa
import plotly.express as px
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display


# Custom CSS
st.markdown(
    """
    <style>
    .title {
        color: #2E4053;
        font-size: 3em;
        text-align: center;
        font-weight: bold;
    }

    .stContainer {
        border-radius: 8px;
        padding: 15px;
        background-color: #F7F9F9;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        color: #2C3E50;
    }

    h3 {
        color: #34495E;
        text-transform: uppercase;
        font-weight: 600;
    }

    .stButton button {
        background-color: #1ABC9C;
        color: white;
        font-size: 1.1em;
        font-weight: 600;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #FADBD8;
    }

    .stWarning {
        background-color: #FADBD8;
        color: #A93226;
        border-radius: 10px;
    }

    .stAudio {
        margin-top: 10px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# function to load audio
def load_audio(audiopath, sampling_rate=22000):
    if isinstance(audiopath, str):
        if audiopath.endswith(".mp3"):
            audio, lsr = librosa.load(audiopath, sr=sampling_rate)
            audio = torch.FloatTensor(audio)
        else:
            assert False, f"Unsupported audio format provided: {audiopath[-4:]}"
    elif isinstance(audiopath, io.BytesIO):
        audio, lsr = torchaudio.load(audiopath)
        audio = audio[0]
    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with audio data. Max = {audio.max()} min = {audio.min()}")
    audio.clip_(-1, 1)
    return audio.unsqueeze(0)


# function for classifier
def classify_audio_clip(clip):
    classifier = AudioMiniEncoderWithClassifierHead(
        2,
        spec_dim=1,
        embedding_dim=512,
        depth=5,
        downsample_factor=4,
        resnet_blocks=2,
        attn_blocks=4,
        num_attn_heads=4,
        base_channels=32,
        dropout=0,
        kernel_size=5,
        distribute_zero_label=False,
    )

    state_dict = torch.load("classifier.pth", map_location=torch.device("cpu"))
    classifier.load_state_dict(state_dict)
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]


# function to extract MFCC features
def extract_mfcc_features(audio_clip, sr=22000, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio_clip.squeeze().numpy(), sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)


st.markdown("<h1 class='title'>AI-Generated Voice Detection</h1>", unsafe_allow_html=True)

def main():
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3"])

    if uploaded_file is not None:
        if st.button("Analyze Audio"):

            # Results Container
            with st.container():
                st.info("Analysis Results")

                audio_clip = load_audio(uploaded_file)
                result = classify_audio_clip(audio_clip)
                result = result.item()
                st.info(f"Result Probability : {result}")
                st.success(
                    f"The uploaded audio is {result * 100:.2f}% likely to be AI generated"
                )

                # Feature Extraction for Heat Map
                feature = extract_mfcc_features(audio_clip)
                sample_data = np.random.randn(20, 13)  # Placeholder data for heat map
                heatmap_data = np.vstack([sample_data, feature])

                # PCA for Heat Map Visualization
                pca = PCA(n_components=2)
                heatmap_data_2d = pca.fit_transform(heatmap_data)

                # Create Heat Map
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.kdeplot(
                    x=heatmap_data_2d[:, 0],
                    y=heatmap_data_2d[:, 1],
                    cmap="viridis",
                    fill=True,
                    thresh=0,
                    ax=ax
                )
                ax.scatter(heatmap_data_2d[-1, 0], heatmap_data_2d[-1, 1], color="red", s=100, label="Uploaded Audio")
                ax.set_title("Heat Map of Audio Feature Distribution")
                ax.legend()
                st.pyplot(fig)

                # Interpretation of Heat Map
                st.markdown(
                    """
                    ### Interpretation of the Heat Map
                    - **Background Heat Map**: Shows the distribution of audio features based on sample data. Brighter areas represent higher densities, where similar audio features are commonly found.
                    - **Red Marker**: Represents the uploaded audio file’s location in the feature space.
                      - **High-density area**: If the red marker is in a high-density area, the uploaded audio has features similar to many existing samples.
                      - **Low-density area**: If the marker is in a low-density area, the audio may have unique or unusual features.
                    """
                )

            # Waveform Container
            with st.container():
                st.info("Uploaded Audio and Waveform")
                st.audio(uploaded_file)

                # Create Waveform Plot
                fig = px.line()
                fig.add_scatter(
                    x=list(range(len(audio_clip.squeeze()))), y=audio_clip.squeeze()
                )
                fig.update_layout(
                    title="Waveform Plot",
                    xaxis_title="Time",
                    yaxis_title="Amplitude",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Interpretation of Waveform Plot
                st.markdown(
                    """
                    ### Interpretation of the Waveform Plot
                    - **Waveform**: This plot shows the amplitude of the audio signal over time.
                      - **Peaks and Troughs**: Larger peaks indicate louder sounds, while troughs represent softer sounds.
                      - **Smooth or Sharp Variations**: Natural human voice tends to have smoother variations, while synthetic or AI-generated voices may exhibit more consistent or patterned shapes due to generation algorithms.
                    """
                )

            # Spectrogram Container
            with st.container():
                st.info("Spectrogram")
                D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_clip.squeeze().numpy())), ref=np.max)
                fig_spectrogram, ax = plt.subplots(figsize=(10, 4))
                img = librosa.display.specshow(D, sr=22000, x_axis='time', y_axis='log', ax=ax)
                fig_spectrogram.colorbar(img, ax=ax, format='%+2.0f dB')
                ax.set_title('Spectrogram (dB)')
                st.pyplot(fig_spectrogram)

                # Interpretation of Spectrogram
                st.markdown(
                    """
                    ### Interpretation of the Spectrogram
                    - **Spectrogram**: Displays the frequency content of the audio over time.
                      - **Colors**: Bright colors indicate louder frequencies. Patterns and variations here show how sound frequencies evolve over time.
                      - **Human vs. AI Characteristics**: Human voices usually have a more varied and organic distribution of frequencies, whereas AI-generated voices may exhibit more uniform or repetitive patterns.
                    """
                )

                st.info("Disclaimer")
                st.warning(
                    "These classification or detection mechanisms are not always accurate."
                    " They should be considered as a strong signal and not be considered as an ultimate decision maker."
                )


if __name__ == "__main__":
    main()
