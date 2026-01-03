"""
Test collator module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

def test_collator_mock():
    """Test collator with mock data"""
    print("=" * 60)
    print("Test 1: Collator Mock Test")
    print("=" * 60)
    
    # Mock processor
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            # Mock encoding for "<|im_start|>assistant\n"
            return [1, 2, 3]
        
        @property
        def pad_token_id(self):
            return 0
    
    class MockProcessor:
        def __init__(self):
            self.tokenizer = MockTokenizer()
    
    from collator import QwenCompletionCollator
    
    processor = MockProcessor()
    collator = QwenCompletionCollator(processor=processor)
    
    print(f"✅ Collator initialized")
    print(f"   Response template: {collator.response_template}")
    print(f"   Response token IDs: {collator.response_token_ids}")
    
    assert collator.response_template == "<|im_start|>assistant\n"
    assert collator.response_token_ids == [1, 2, 3]
    print("✅ Template tokens OK\n")

def test_label_masking():
    """Test label masking logic"""
    print("=" * 60)
    print("Test 2: Label Masking")
    print("=" * 60)
    
    # Create mock collator
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [10, 20, 30]  # Mock response template tokens
        
        @property
        def pad_token_id(self):
            return 0
    
    class MockProcessor:
        def __init__(self):
            self.tokenizer = MockTokenizer()
    
    from collator import QwenCompletionCollator
    
    processor = MockProcessor()
    collator = QwenCompletionCollator(processor=processor)
    
    # Create mock input_ids and labels
    # Format: [system tokens] [user tokens] [10, 20, 30] [assistant response tokens]
    input_ids = torch.tensor([
        [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60],  # Has template at position 5-7
        [1, 2, 3, 10, 20, 30, 40, 50, 0, 0, 0],   # Has template at position 3-5, with padding
    ])
    
    labels = input_ids.clone()
    
    # Apply masking
    masked_labels = collator._mask_labels(input_ids, labels)
    
    print("Input IDs:")
    print(input_ids)
    print("\nMasked Labels:")
    print(masked_labels)
    
    # Verify masking
    # First sequence: mask positions 0-7 (before and including template)
    assert (masked_labels[0, :8] == -100).all(), "Should mask before assistant response"
    assert (masked_labels[0, 8:] != -100).any(), "Should not mask assistant response"
    
    # Second sequence: mask positions 0-5
    assert (masked_labels[1, :6] == -100).all(), "Should mask before assistant response"
    
    print("\n✅ Label masking works correctly\n")

def test_format_specification():
    """Verify the output format specification"""
    print("=" * 60)
    print("Test 3: Format Specification Verification")
    print("=" * 60)
    
    print("Expected assistant response format:")
    print("  <think_text> {answer: \"value\"}")
    print()
    print("Examples:")
    print("  1. 'Looking at the chart, value is 42. {answer: \"42\"}'")
    print("  2. 'Step 1... Step 2... {answer: \"100\"}'")
    print("  3. '{answer: \"yes\"}' (if no think text)")
    print()
    print("✅ Format specification documented\n")

if __name__ == "__main__":
    print("\n🧪 Running Collator Tests\n")
    
    try:
        test_collator_mock()
        test_label_masking()
        test_format_specification()
        
        print("=" * 60)
        print("🎉 All collator tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
