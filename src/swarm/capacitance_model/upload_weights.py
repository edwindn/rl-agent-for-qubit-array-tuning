import torch
from huggingface_hub import HfApi
from CapacitancePrediction import create_model

repo_id = "edwindn/capacitance_barriers_model"
weights_path = "weights/best_model_barriers.pth"
output_size = 2
mobilenet = "small"

model = create_model(output_size=output_size, mobilenet=mobilenet)

checkpoint = torch.load(weights_path, map_location='cpu')

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

api = HfApi()
api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True,
)

torch.save(model, "model.pth")
torch.save(state_dict, "weights.pth")

api.upload_file(
    path_or_fileobj="model.pth",
    path_in_repo="model.pth",
    repo_id=repo_id,
    repo_type="model",
)

api.upload_file(
    path_or_fileobj="weights.pth",
    path_in_repo="weights.pth",
    repo_id=repo_id,
    repo_type="model",
)

print(f"Model and weights uploaded successfully to {repo_id}")

