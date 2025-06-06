# -*- coding: utf-8 -*-
"""Call Quality Metrics Analysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IPeyh5o3HN0oAy9KF8BTUG7tD3yaUmpB
"""

import os
import shutil

upload_dir = "All_Conversations"
os.makedirs(upload_dir, exist_ok=True)

# Move all .json files into the folder
for fname in os.listdir():
    if fname.endswith(".json"):
        shutil.move(fname, os.path.join(upload_dir, fname))

print(f"✅ Moved all .json files to {upload_dir}/")

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to calculate overtalk and silence percentage
def calculate_overtalk_silence_percentage(convo):
    total_duration = 0
    overtalk_duration = 0
    silence_duration = 0

    last_speaker = None
    last_end_time = None

    for utt in convo:
        start_time = utt['stime']
        end_time = utt['etime']
        speaker = utt['speaker']

        # Update total conversation time
        total_duration += (end_time - start_time)

        if last_speaker is not None and last_end_time is not None:
            # Check for overtalk
            overlap_time = max(0, min(last_end_time, end_time) - max(last_start_time, start_time))
            if overlap_time > 0:
                overtalk_duration += overlap_time

        # Check for silence (gap between two consecutive utterances)
        if last_end_time is not None and start_time > last_end_time:
            silence_duration += (start_time - last_end_time)

        last_speaker = speaker
        last_start_time = start_time
        last_end_time = end_time

    # Calculate percentages
    overtalk_percentage = (overtalk_duration / total_duration) * 100 if total_duration > 0 else 0
    silence_percentage = (silence_duration / total_duration) * 100 if total_duration > 0 else 0

    return overtalk_percentage, silence_percentage

# Iterate over the files to calculate overtalk and silence for each conversation
overtalk_percentages = []
silence_percentages = []
call_ids = []

folder_path = 'All_Conversations'  # Change this to your folder path

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            convo = json.load(f)

        overtalk, silence = calculate_overtalk_silence_percentage(convo)
        overtalk_percentages.append(overtalk)
        silence_percentages.append(silence)
        call_ids.append(filename)

# Visualization
df = {
    'Call ID': call_ids,
    'Overtalk Percentage': overtalk_percentages,
    'Silence Percentage': silence_percentages
}

# Create a DataFrame
import pandas as pd
df = pd.DataFrame(df)

# Set up the plot
plt.figure(figsize=(12, 6))

# Plot Overtalk vs Silence
sns.set(style="whitegrid")
plt.barh(df['Call ID'], df['Overtalk Percentage'], label='Overtalk', color='orange', alpha=0.7)
plt.barh(df['Call ID'], df['Silence Percentage'], label='Silence', color='blue', alpha=0.7)
plt.xlabel('Percentage (%)')
plt.title('Overtalk vs Silence Percentage per Call')

# Add legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()