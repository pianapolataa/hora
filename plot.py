import matplotlib.pyplot as plt
import numpy as np

# If you have a file, use: data = np.loadtxt('trajectory.txt')
# Otherwise, paste your lines into this list:
data = np.array([
[-0.17000000178813934, -0.07000000029802322, 0.4699999988079071, -0.17422400414943695, -0.05785555765032768, 0.4995972216129303, -0.1756303310394287, -0.053023435175418854, 0.488307386636734, -0.18367350101470947, -0.04203217476606369, 0.4545298218727112, -0.18619339168071747, -0.04600166901946068, 0.4646144509315491, -0.13243800401687622, -0.08989059925079346, 0.4962785243988037]
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