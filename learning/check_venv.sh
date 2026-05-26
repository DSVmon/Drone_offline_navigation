#!/bin/bash
source /mnt/e/Git_store/learning/venv/bin/activate
python3 -c 'import gymnasium; import stable_baselines3; import torch; print("all packages OK")'
echo "Python: $(which python3)"
echo "gymnasium: $(python3 -c 'import gymnasium; print(gymnasium.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "sb3: $(python3 -c 'import stable_baselines3; print(stable_baselines3.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "torch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "tensorboard: $(python3 -c 'import tensorboard; print(tensorboard.__version__)' 2>/dev/null || echo 'NOT FOUND')"
