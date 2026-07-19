---
description: Verify Python syntax for learning/ files. Usage: verify-syntax [file1.py file2.py ...] or verify-syntax --all for all learning files.
---

# Syntax Verification

Verify Python syntax for the specified learning files, or all learning files if --all is passed.

## Usage

```
verify-syntax --all                    # Check all learning/ Python files
verify-syntax drone_env.py reward.py   # Check specific files
```

## Implementation

Run from project root `/mnt/e/Git_store/Drone_offline_navigation`:

### All files

```bash
cd /mnt/e/Git_store/Drone_offline_navigation && python3 -c "
import ast, sys
files = [
    'learning/config.py', 'learning/drone_env.py', 'learning/train.py',
    'learning/reward.py', 'learning/curriculum.py', 'learning/callbacks.py',
    'learning/policy.py', 'learning/vae.py', 'learning/recurrent_ppo.py',
    'learning/collect_data.py', 'learning/inference_node.py', 'learning/utils.py'
]
ok = True
for f in files:
    try:
        with open(f) as fh:
            ast.parse(fh.read())
        print(f'{f}: OK')
    except FileNotFoundError:
        print(f'{f}: SKIP (not found)')
    except SyntaxError as e:
        print(f'{f}: ERROR - {e}')
        ok = False
if ok:
    print('All files: Syntax OK')
else:
    print('ERRORS found!')
    sys.exit(1)
"
```

### Single file

```bash
cd /mnt/e/Git_store/Drone_offline_navigation && python3 -m py_compile learning/$1 && echo "OK"
```
