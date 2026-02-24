import matplotlib.pyplot as plt
import numpy as np

# If you have a file, use: data = np.loadtxt('trajectory.txt')
# Otherwise, paste your lines into this list:
data = np.array([
[0.009999999776482582, 0.0, 0.6000000238418579, -0.18007497489452362, -0.08009221404790878, 0.5167308449745178, -0.192018523812294, -0.06986267864704132, 0.5054121017456055, -0.19033870100975037, -0.06458662450313568, 0.4744791090488434, -0.1834591180086136, -0.07649197429418564, 0.4811698794364929, -0.12569274008274078, -0.10199690610170364, 0.5078871250152588]
])

def plot_grasp_frame(frame_data):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Extract Ball (Indices 0, 1, 2)
    ball = frame_data[0:3]
    ax.scatter(ball[0], ball[1], ball[2], color='red', s=200, label='Object', marker='o')

    # 2. Extract Fingertips (Indices 3 to 17)
    # We reshape the remaining 15 values into (5, 3)
    tips = frame_data[3:].reshape(-1, 3)
    tip_labels = ['Index', 'Middle', 'Ring', 'Pinky', 'Thumb']
    colors = ['blue', 'green', 'purple', 'orange', 'cyan']

    for i in range(len(tips)):
        ax.scatter(tips[i, 0], tips[i, 1], tips[i, 2], color=colors[i], s=100, label=tip_labels[i])
        # Draw a line from ball to finger to visualize the "Gap"
        ax.plot([ball[0], tips[i, 0]], [ball[1], tips[i, 1]], [ball[2], tips[i, 2]], 'k--', alpha=0.3)

    # Axis labels & Limits
    ax.set_xlabel('X (Lateral)')
    ax.set_ylabel('Y (Forward)')
    ax.set_zlabel('Z (Height)')
    ax.set_title('Fingertip vs Object Relative Positions')
    
    # Force the view to stay consistent (adjust based on your coords)
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([0.0, 0.4])
    ax.set_zlim([0.4, 0.8])
    
    ax.legend()
    plt.show()

# To plot the very last frame recorded:
if len(data) > 0:
    plot_grasp_frame(data[-1])
else:
    print("No data found. Paste your printed arrays into the 'data' list.")