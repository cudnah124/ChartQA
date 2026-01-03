"""
Test data loader module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import format_response, build_conversation, load_json_data
from config import get_config

def test_format_response():
    """Test response formatting"""
    print("=" * 60)
    print("Test 1: Response Formatting")
    print("=" * 60)
    
    # Test with think and answer
    think = "Looking at the chart, the value is 42."
    answer = "42"
    result = format_response(think, answer)
    
    print(f"Think: {think}")
    print(f"Answer: {answer}")
    print(f"Result: {result}")
    
    assert think in result
    assert '{answer: "42"}' in result
    assert result.endswith('{answer: "42"}')
    print("✅ Format with think OK\n")
    
    # Test with empty think
    result2 = format_response("", "100")
    print(f"Empty think result: {result2}")
    assert result2 == '{answer: "100"}'
    print("✅ Format without think OK\n")

def test_build_conversation():
    """Test conversation building"""
    print("=" * 60)
    print("Test 2: Conversation Building")
    print("=" * 60)
    
    config = get_config()
    system_prompt = config.system_prompt.prompt
    
    conversation = build_conversation(
        system_prompt=system_prompt,
        question="What is the value?",
        think="The chart shows 42",
        label="42"
    )
    
    print(f"System: {conversation[0]['content'][:50]}...")
    print(f"User: {conversation[1]['content']}")
    print(f"Assistant: {conversation[2]['content']}")
    
    # Verify structure
    assert len(conversation) == 3
    assert conversation[0]['role'] == 'system'
    assert conversation[1]['role'] == 'user'
    assert conversation[2]['role'] == 'assistant'
    
    # Verify content
    assert "step-by-step" in conversation[0]['content']
    assert conversation[1]['content'] == "What is the value?"
    assert "The chart shows 42" in conversation[2]['content']
    assert '{answer: "42"}' in conversation[2]['content']
    
    print("✅ Conversation structure OK\n")

def test_load_data():
    """Test data loading"""
    print("=" * 60)
    print("Test 3: Data Loading")
    print("=" * 60)
    
    config = get_config()
    
    if os.path.exists(config.data.train_file):
        data = load_json_data(config.data.train_file)
        print(f"✅ Loaded {len(data)} items")
        
        # Check first item structure
        if len(data) > 0:
            item = data[0]
            required_fields = ['image', 'question', 'think', 'answer', 'label']
            for field in required_fields:
                assert field in item, f"Missing field: {field}"
            print(f"✅ Data structure OK")
            
            print(f"\nSample item:")
            print(f"  Image: {item['image']}")
            print(f"  Question: {item['question'][:50]}...")
            print(f"  Think: {item['think'][:50]}...")
            print(f"  Answer: {item['answer']}")
            print(f"  Label: {item['label']}")
    else:
        print(f"⚠️  Data file not found: {config.data.train_file}")
        print("   (This is OK if data hasn't been prepared yet)")
    
    print()

def test_response_format_spec():
    """Test that response format matches specification"""
    print("=" * 60)
    print("Test 4: Response Format Specification")
    print("=" * 60)
    
    # Specification: think_text + {answer: "value"}
    think = "Step 1: Read the chart. Step 2: Find the value."
    answer = "19.5"
    
    result = format_response(think, answer)
    
    print(f"Expected format: <think_text> {{answer: \"value\"}}")
    print(f"Actual result: {result}")
    
    # Verify format
    parts = result.split('{answer:')
    assert len(parts) == 2, "Should have think text and answer JSON"
    assert parts[0].strip() == think, "Think text should be preserved"
    assert '{answer: "19.5"}' in result, "Answer should be in JSON format"
    
    print("✅ Format matches specification\n")

if __name__ == "__main__":
    print("\n🧪 Running Data Loader Tests\n")
    
    try:
        test_format_response()
        test_build_conversation()
        test_load_data()
        test_response_format_spec()
        
        print("=" * 60)
        print("🎉 All data loader tests passed!")
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
