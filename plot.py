import matplotlib.pyplot as plt
import numpy as np

# If you have a file, use: data = np.loadtxt('trajectory.txt')
# Otherwise, paste your lines into this list:
data = np.array([
[-0.17000000178813934, -0.07000000029802322, 0.5699999928474426, -0.18196307122707367, -0.059685252606868744, 0.5299776196479797, -0.1871613711118698, -0.05235975980758667, 0.5225902199745178, -0.18164204061031342, -0.05873025208711624, 0.48413223028182983, -0.1904723048210144, -0.05567392706871033, 0.49822020530700684, -0.12842020392417908, -0.08565885573625565, 0.5042527914047241]
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