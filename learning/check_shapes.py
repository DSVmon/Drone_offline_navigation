import torch
import sys
sys.path.insert(0, "/mnt/e/Git_store/Drone_offline_navigation/learning")
import config

from bc_model import BCNet
bc = BCNet(20, 3)
bc.load_state_dict(torch.load("expert_data/bc_policy.pt", map_location="cpu"))

print("BC model layers:")
for name, p in bc.state_dict().items():
    print(f"  {name}: {p.shape}")

print(f"\nBC_HIDDEN = {config.BC_HIDDEN}")
print(f"PPO net_arch = {config.POLICY_KWARGS['net_arch']}")
