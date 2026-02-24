import matplotlib.pyplot as plt
import numpy as np

# If you have a file, use: data = np.loadtxt('trajectory.txt')
# Otherwise, paste your lines into this list:
data = np.array([
[-0.17000000178813934, -0.07000000029802322, 0.5699999928474426, -0.18895068764686584, -0.04708591848611832, 0.4901217520236969, -0.18875882029533386, -0.04308536648750305, 0.47954440116882324, -0.18270555138587952, -0.035735130310058594, 0.44852152466773987, -0.18759402632713318, -0.038823992013931274, 0.452525794506073, -0.12974052131175995, -0.07815016061067581, 0.47554680705070496]
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