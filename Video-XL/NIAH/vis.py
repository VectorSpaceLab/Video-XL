import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import os
import glob
import argparse
import pdb


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='tmp-outputs/', type=str, metavar='N', help='which model to use')
    parser.add_argument('--results_dir', default='/share/minghao/Projects/NeedleInAVideoHaystack/newresults')
    args = parser.parse_args()
    return args


args = get_args()

# Path to the directory containing JSON results
folder_path = os.path.join(args.results_dir, args.model_name)

# Using glob to find all json files in the directory
json_files = glob.glob(f"{folder_path}/*.json")

# List to hold the data
data = []

avg_score = 0
avg_cnt = 0
all_depth = []
# Iterating through each file and extract the 3 columns we need
for file in json_files:
    with open(file, 'r') as f:
        json_data = json.load(f)
        # Extracting the required fields
        video_depth = json_data.get("depth_percent", None)

        if video_depth not in all_depth:
            all_depth.append(video_depth)

        context_length = json_data.get("context_length", None)
        score = json_data.get("score", None)
        # Appending to the list
        data.append({
            "Video Depth": video_depth,
            "Context Length": context_length,
            "Score": score
        })
        avg_score += score
        avg_cnt += 1
print(avg_score / avg_cnt)

# all_depth.sort()
# pdb.set_trace()
# for depth in all_depth:
#     data.insert(0, {
#             "Video Depth": depth,
#             "Context Length": 1,
#             "Score": 1.0
#         })

# Creating a DataFrame
df = pd.DataFrame(data)

print (df.head())
print (f"You have {len(df)} rows")

pivot_table = pd.pivot_table(df, values='Score', index=['Video Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
pivot_table = pivot_table.pivot(index="Video Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
print(pivot_table)
pivot_table.iloc[:5, :5]
print(pivot_table)

# Create a custom colormap. Go to https://coolors.co/ and pick cool colors
cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
# cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#DD576F", "#E1A457", "#BCC889", "#A6D3B5"])

# Create the heatmap with better aesthetics
plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
sns.set(font_scale=2.2)  # 调整全局字体大小

# Create the heatmap with better aesthetics
plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
sns.heatmap(
    pivot_table,
    # annot=True,
    fmt="g",
    cmap=cmap,
    cbar_kws={'label': 'Score'},
    linewidths=1.5,
    vmin=0,   # 将颜色范围的下界设为0
    vmax=1    # 将颜色范围的上界设为1
)

# More aesthetics
# plt.title('')  # Adds a title
plt.xlabel('Num. Frame')  # X-axis label
plt.ylabel('Frame Depth')  # Y-axis label
plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
plt.tight_layout()  # Fits everything neatly into the figure area

# Show the plot
# plt.show()
save_dir = os.path.join(args.results_dir, 'img')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(f"{save_dir}/{args.model_name}.png")