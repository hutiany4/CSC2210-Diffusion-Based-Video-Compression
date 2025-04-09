import os

def get_size(path):
    """
    Returns the total size (in bytes) of a file or folder.
    """
    if os.path.isfile(path):
        # It's a file, just return its size
        return os.path.getsize(path)
    else:
        # It's a folder; walk through all subfiles/subfolders
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size

def calculate_compression_rate(video_path, keyframes_path, canny_path, frames_npy_path, h265_file_path):
    """
    Calculate the compression rate by:
      (size of all encoded files) / (size of original video)
    """
    original_video_size = get_size(video_path)
    encoded_size = (get_size(keyframes_path) 
                    + get_size(canny_path)
                    + get_size(frames_npy_path))
    h265_encoded_size = get_size(h265_file_path)
    
    if original_video_size == 0:
        raise ValueError("Original video size is 0 bytes. Check the path.")
    
    print(f"Encoded size: {encoded_size:.4f}")
    print(f"original video size: {original_video_size:.4f}")
    print(f"h265 video size: {h265_encoded_size:.4f}")
    
    ours_compression_rate = encoded_size / original_video_size
    h265_compression_rate = h265_encoded_size / original_video_size
    return ours_compression_rate, h265_compression_rate

if __name__ == "__main__":
    filename = "road"
    # Replace these paths with your actual file/folder paths
    video_path = f"./output/{filename}/raw/"
    keyframes_folder = f"./output/{filename}/keyframes/"
    canny_folder = f"./output/{filename}/auxiliary/"
    frames_npy_path = f"./output/{filename}/frames.npy"
    h265_file_path = f"./output/{filename}/h265_{filename}.mp4"
    
    rate1, rate2 = calculate_compression_rate(video_path, keyframes_folder, canny_folder, frames_npy_path, h265_file_path)
    
    print(f"Ours Compression rate: {rate1:.4f}")
    print(f"H265 Compression rate: {rate2:.4f}")
