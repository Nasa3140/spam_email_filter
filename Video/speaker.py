from moviepy.editor import VideoFileClip
from pyAudioAnalysis import audioSegmentation
import matplotlib.pyplot as plt


def extract_audio_from_video(video_file, audio_file):
    # Load the video file
    video = VideoFileClip(video_file)

    # Extract the audio from the video
    audio = video.audio

    # Save the audio as an audio file
    audio.write_audiofile(audio_file)
    print("Audio file created successfully:", audio_file)

    # Close the video file
    video.close()


def perform_speaker_diarization(audio_file, n_speakers):
    # Perform speaker diarization
    labels = audioSegmentation.speaker_diarization(audio_file, n_speakers=n_speakers, mid_window=2, mid_step=1)

    # Empty list to collect speakers count
    li=[]

    # Calculate the number of speakers
    speaker_labels = [li.append(label) for label in range(len(labels)) if label != 0]
    num_speakers = len(speaker_labels)

    print("Number of speakers:", li)
    print("Total speakers:", num_speakers)

    # Plot the speaker diarization segments
    plt.figure(figsize=(10, 4))
    for i, label in enumerate(li):
        plt.plot([i, i+1], [label, label], color='blue')
    plt.xlabel('Segment')
    plt.ylabel('Speaker')
    plt.title('Speaker Diarization')
    plt.show()


def main():
    video_file = "ratantata.mp4"
    audio_file = "ratantata.wav"
    n_speakers = 2  # Specify the expected number of speakers

    extract_audio_from_video(video_file, audio_file)
    print("Audio extraction completed.\n")

    print("Performing speaker diarization:")
    perform_speaker_diarization(audio_file, n_speakers)


if __name__ == "__main__":
    main()
