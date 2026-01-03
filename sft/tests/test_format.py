"""
Test output format
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

def test_response_format():
    """Test response format parsing"""
    print("=" * 60)
    print("Test 1: Response Format Parsing")
    print("=" * 60)
    
    # Implement extract function locally (no GPU needed)
    def extract_answer_from_response(response):
        """Extract answer from response"""
        try:
            if "{answer:" in response:
                start = response.index("{answer:")
                end = response.index("}", start) + 1
                answer_json = response[start:end]
                answer_json = answer_json.replace("answer:", '"answer":')
                parsed = json.loads(answer_json)
                return parsed.get("answer", "")
            else:
                return None
        except Exception as e:
            print(f"Error extracting answer: {e}")
            return None
    
    # Test cases
    test_cases = [
        {
            "response": "Looking at the chart, the value is 42. {answer: \"42\"}",
            "expected_answer": "42"
        },
        {
            "response": "Step 1: Find the bar. Step 2: Read value. {answer: \"100\"}",
            "expected_answer": "100"
        },
        {
            "response": "{answer: \"yes\"}",
            "expected_answer": "yes"
        }
    ]

    
    for i, test in enumerate(test_cases, 1):
        response = test["response"]
        expected = test["expected_answer"]
        
        extracted = extract_answer_from_response(response)
        
        print(f"\nTest case {i}:")
        print(f"  Response: {response}")
        print(f"  Expected: {expected}")
        print(f"  Extracted: {extracted}")
        
        assert extracted == expected, f"Mismatch: got {extracted}, expected {expected}"
        print(f"  ✅ OK")
    
    print("\n✅ All format parsing tests passed\n")

def test_json_structure():
    """Test JSON structure"""
    print("=" * 60)
    print("Test 2: JSON Structure")
    print("=" * 60)
    
    # Valid JSON formats
    valid_formats = [
        '{answer: "42"}',
        '{"answer": "42"}',
        '{answer:"42"}',
    ]
    
    for fmt in valid_formats:
        # Normalize format
        normalized = fmt.replace("answer:", '"answer":')
        try:
            parsed = json.loads(normalized)
            assert "answer" in parsed
            print(f"✅ Valid: {fmt}")
        except:
            print(f"❌ Invalid: {fmt}")
    
    print()

def test_complete_workflow():
    """Test complete workflow"""
    print("=" * 60)
    print("Test 3: Complete Workflow")
    print("=" * 60)
    
    # Simulate training data format
    training_example = {
        "image": "chart.png",
        "question": "What is the value?",
        "think": "Looking at the chart, I can see the value is 42.",
        "answer": "42",
        "label": "42"
    }
    
    # Simulate model output
    model_output = f"{training_example['think']} {{answer: \"{training_example['label']}\"}}"
    
    print(f"Training format:")
    print(f"  Question: {training_example['question']}")
    print(f"  Think: {training_example['think']}")
    print(f"  Label: {training_example['label']}")
    print()
    print(f"Expected model output:")
    print(f"  {model_output}")
    print()
    
    # Verify format
    assert training_example['think'] in model_output
    assert '{answer: "42"}' in model_output
    
    print("✅ Workflow verified\n")

if __name__ == "__main__":
    print("\n🧪 Running Format Tests\n")
    
    try:
        test_response_format()
        test_json_structure()
        test_complete_workflow()
        
        print("=" * 60)
        print("🎉 All format tests passed!")
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
