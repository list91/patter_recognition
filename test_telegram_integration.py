"""
Test script to verify telegram_logger integration
"""

import telegram_logger
from pathlib import Path

print("="*60)
print("Testing Telegram Logger Integration")
print("="*60)

# Test 1: Module import
print("\n1. Testing module import...")
print("   [OK] Module imported successfully")

# Test 2: Mock trainer callback
print("\n2. Testing callback with mock trainer...")

class MockTrainer:
    """Mock YOLO trainer object for testing"""
    def __init__(self):
        self.save_dir = Path('runs/detect/scheme_detector')

trainer = MockTrainer()

try:
    telegram_logger.on_epoch_end(trainer)
    print("   [OK] Callback executed successfully")
except Exception as e:
    print(f"   [ERROR] Callback failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Verify imports work in training scripts
print("\n3. Testing imports in training scripts...")

try:
    # Test train.py
    with open('train.py', 'r', encoding='utf-8') as f:
        content = f.read()
        if 'import telegram_logger' in content:
            print("   [OK] train.py has telegram_logger import")
        else:
            print("   [ERROR] train.py missing telegram_logger import")

        if 'model.add_callback("on_train_epoch_end", telegram_logger.on_epoch_end)' in content:
            print("   [OK] train.py has callback registration")
        else:
            print("   [ERROR] train.py missing callback registration")

    # Test train_lowmem.py
    with open('train_lowmem.py', 'r', encoding='utf-8') as f:
        content = f.read()
        if 'import telegram_logger' in content:
            print("   [OK] train_lowmem.py has telegram_logger import")
        else:
            print("   [ERROR] train_lowmem.py missing telegram_logger import")

        if 'model.add_callback("on_train_epoch_end", telegram_logger.on_epoch_end)' in content:
            print("   [OK] train_lowmem.py has callback registration")
        else:
            print("   [ERROR] train_lowmem.py missing callback registration")

except Exception as e:
    print(f"   [ERROR] Failed to verify training scripts: {e}")

# Test 4: Check .gitignore
print("\n4. Testing .gitignore configuration...")
try:
    with open('.gitignore', 'r', encoding='utf-8') as f:
        content = f.read()
        if 'mAP_progress.png' in content:
            print("   [OK] .gitignore includes mAP_progress.png")
        else:
            print("   [ERROR] .gitignore missing mAP_progress.png")
except Exception as e:
    print(f"   [ERROR] Failed to check .gitignore: {e}")

print("\n" + "="*60)
print("Integration Test Complete")
print("="*60)
