"""
Test configuration module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, get_config

def test_config_creation():
    """Test basic config creation"""
    print("=" * 60)
    print("Test 1: Config Creation")
    print("=" * 60)
    
    config = get_config()
    
    # Check model config
    assert config.model.model_name == "unsloth/Qwen2-VL-2B-Instruct"
    assert config.model.max_seq_length == 1024
    assert config.model.load_in_4bit == True
    print("✅ Model config OK")
    
    # Check training config
    assert config.training.per_device_train_batch_size == 2
    assert config.training.gradient_accumulation_steps == 4
    assert config.training.learning_rate == 2e-5
    print("✅ Training config OK")
    
    # Check system prompt
    assert "step-by-step reasoning" in config.system_prompt.prompt
    assert "{answer:" in config.system_prompt.prompt
    print("✅ System prompt OK")
    
    print("\n✅ All config tests passed!\n")

def test_config_paths():
    """Test data paths"""
    print("=" * 60)
    print("Test 2: Data Paths")
    print("=" * 60)
    
    config = get_config()
    
    print(f"Train file: {config.data.train_file}")
    print(f"Val file: {config.data.val_file}")
    print(f"Train images: {config.data.train_images}")
    print(f"Val images: {config.data.val_images}")
    
    # Check paths are absolute
    assert os.path.isabs(config.data.train_file)
    assert os.path.isabs(config.data.val_file)
    print("\n✅ Paths are absolute!")
    
    # Check if files exist
    if os.path.exists(config.data.train_file):
        print("✅ Train file exists")
    else:
        print("⚠️  Train file not found (expected if not yet created)")
    
    print()

def test_config_dict_conversion():
    """Test config to/from dict conversion"""
    print("=" * 60)
    print("Test 3: Dict Conversion")
    print("=" * 60)
    
    config = get_config()
    config_dict = config.to_dict()
    
    assert 'model' in config_dict
    assert 'training' in config_dict
    assert 'system_prompt' in config_dict
    print("✅ Config to dict conversion OK")
    
    # Test from dict
    new_config = Config.from_dict(config_dict)
    assert new_config.model.model_name == config.model.model_name
    assert new_config.training.learning_rate == config.training.learning_rate
    print("✅ Dict to config conversion OK\n")

if __name__ == "__main__":
    print("\n🧪 Running Configuration Tests\n")
    
    try:
        test_config_creation()
        test_config_paths()
        test_config_dict_conversion()
        
        print("=" * 60)
        print("🎉 All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
