import torch

original_ckpt_path = "/w/339/frankli/CSC2210-Project/CSC2210-Diffusion-Based-Video-Compression/models/control_sd15_canny.pth"
my_ckpt_path = "/w/339/frankli/CSC2210-Project/CSC2210-Diffusion-Based-Video-Compression/models/ours_sd15_canny.pth"
merged_ckpt_path = "/w/339/frankli/CSC2210-Project/CSC2210-Diffusion-Based-Video-Compression/models/new_control_sd15_canny.pth"

# 1. Load original checkpoint
orig_ckpt = torch.load(original_ckpt_path, map_location="cpu")
# Some checkpoints have the weights under ["state_dict"], others are top-level
if "state_dict" in orig_ckpt:
    orig_sd = orig_ckpt["state_dict"]
else:
    orig_sd = orig_ckpt  # If everything is directly in the top-level dict

# 2. Load your fine-tuned ControlNet checkpoint
my_ckpt = torch.load(my_ckpt_path, map_location="cpu")
if "state_dict" in my_ckpt:
    my_sd = my_ckpt["state_dict"]
else:
    my_sd = my_ckpt

# 3. Replace any weight in the original that starts with "control_model."
total_line_change = 0
for k, v in my_sd.items():
    curr_key = "control_model." + k
    if curr_key in orig_sd:
        total_line_change += 1
        print(f"Convert value of {curr_key} from {orig_sd[curr_key]} to {v}")
        orig_sd[curr_key] = v

print(f"There are totally {total_line_change} lines changed!")
# 4. Save the merged checkpoint
#   (If original_ckpt had a structure like {"state_dict": ...}, keep that structure)
if "state_dict" in orig_ckpt:
    orig_ckpt["state_dict"] = orig_sd
    torch.save(orig_ckpt, merged_ckpt_path)
else:
    torch.save(orig_sd, merged_ckpt_path)

print(f"Done! Merged checkpoint saved to {merged_ckpt_path}")
