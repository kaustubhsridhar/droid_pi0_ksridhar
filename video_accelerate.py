import os
import subprocess

def accelerate_videos(input_folder, output_folder, speed=5.0):
    """Accelerates all MP4 videos in the input folder and saves them in the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"accelerated_{filename}")
            
            command = [
                "ffmpeg", "-i", input_path, "-filter:v", f"setpts=1/{speed}*PTS", "-an", output_path
            ]
            
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Processed: {filename} -> {output_path}")

if __name__ == "__main__":
    input_folder = "/home/franka/droid_pi0/demo_videos/Tony's PPT"  # Change this to your input folder
    output_folder = "/home/franka/droid_pi0/demo_videos/TONY_Accelerated"  # Change this to your output folder
    
    accelerate_videos(input_folder, output_folder)
