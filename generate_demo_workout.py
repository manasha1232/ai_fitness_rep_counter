import os
import cv2
import numpy as np

def draw_synthetic_squat_person(img, center_x, ground_y, squat_depth_ratio=0.0):
    """
    Draws a synthetic human stickman pose performing a squat.
    squat_depth_ratio: 0.0 = Standing straight, 1.0 = Deep squat.
    """
    color = (240, 240, 240)
    joint_col = (0, 255, 120)
    
    # Standing dimensions
    torso_len = 120
    leg_len = 130
    
    # Calculate hip, knee, ankle positions based on squat depth
    # As person squats, hip drops down, knee moves forward/bends
    hip_y = ground_y - leg_len + int(squat_depth_ratio * 55)
    hip = (center_x, hip_y)
    
    # Head & Shoulder
    head = (center_x, hip_y - torso_len - 30)
    shoulder = (center_x, hip_y - torso_len)
    
    knee_offset_x = int(squat_depth_ratio * 40)
    knee_left = (center_x - 20 - knee_offset_x, hip_y + int(leg_len * 0.5))
    knee_right = (center_x + 20 + knee_offset_x, hip_y + int(leg_len * 0.5))
    
    ankle_left = (center_x - 30, ground_y)
    ankle_right = (center_x + 30, ground_y)
    
    # Arms extended forward for balance
    arm_ext_x = int(50 + squat_depth_ratio * 20)
    elbow_left = (center_x - arm_ext_x // 2, shoulder[1] + 20)
    wrist_left = (center_x - arm_ext_x, shoulder[1] + 15)
    
    elbow_right = (center_x + arm_ext_x // 2, shoulder[1] + 20)
    wrist_right = (center_x + arm_ext_x, shoulder[1] + 15)
    
    # Draw Head
    cv2.circle(img, head, 25, color, -1)
    cv2.circle(img, head, 25, (0, 230, 255), 2)
    
    # Draw Torso
    cv2.line(img, shoulder, hip, color, 6)
    
    # Draw Left Arm
    cv2.line(img, shoulder, elbow_left, color, 4)
    cv2.line(img, elbow_left, wrist_left, color, 4)
    
    # Draw Right Arm
    cv2.line(img, shoulder, elbow_right, color, 4)
    cv2.line(img, elbow_right, wrist_right, color, 4)
    
    # Draw Left Leg
    cv2.line(img, hip, knee_left, color, 5)
    cv2.line(img, knee_left, ankle_left, color, 5)
    
    # Draw Right Leg
    cv2.line(img, hip, knee_right, color, 5)
    cv2.line(img, knee_right, ankle_right, color, 5)
    
    # Draw Joint Nodes
    joints = [head, shoulder, hip, knee_left, knee_right, ankle_left, ankle_right, elbow_left, wrist_left, elbow_right, wrist_right]
    for j in joints:
        cv2.circle(img, j, 6, joint_col, -1)

def generate_synthetic_workout_video(output_path="input/sample_workout_squats.mp4", num_frames=180, width=800, height=600, fps=30):
    """
    Generates a synthetic workout video showing 4 complete squat repetitions.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        output_path = output_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
    print(f"[+] Generating synthetic squat workout video: '{output_path}' ({num_frames} frames)...")
    
    bg = np.ones((height, width, 3), dtype=np.uint8) * 35
    # Add gym floor line
    cv2.line(bg, (50, 520), (width - 50, 520), (80, 80, 80), 3)
    
    for f in range(num_frames):
        frame = bg.copy()
        
        # 4 Squat cycles over 180 frames (cycle period = 45 frames)
        cycle_phase = (f % 45) / 45.0
        # Sinusoidal motion: 0.0 (standing) -> 1.0 (deep squat) -> 0.0 (standing)
        squat_depth = np.sin(cycle_phase * np.pi)
        
        draw_synthetic_squat_person(frame, width // 2, 510, squat_depth_ratio=squat_depth)
        
        # Header text
        cv2.putText(frame, f"AI FITNESS SQUAT BENCHMARK | FRAME: {f+1:03d}/{num_frames}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)
                    
        out.write(frame)
        
    out.release()
    print(f"[OK] Synthetic workout video created at '{output_path}'")
    return output_path

if __name__ == "__main__":
    generate_synthetic_workout_video()
