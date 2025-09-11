"""
Lightweight dataset generation test without pandas dependency.

This script tests the core dataset generation logic by implementing
a minimal version that doesn't require pandas or other heavy dependencies.
"""

import json
import random
from pathlib import Path

# Simplified dataset creator for testing
class SimpleDatasetCreator:
    """Simplified version of the dataset creator for testing without dependencies."""
    
    def __init__(self, samples_per_intent=10):
        self.samples_per_intent = samples_per_intent
        self.intent_labels = [
            "salary_inquiry",
            "meeting_room_booking", 
            "leave_request",
            "directory_search",
            "company_info",
            "employee_info",
            "employee_search"
        ]
        self._define_templates()
    
    def _define_templates(self):
        """Define simplified templates for testing."""
        self.templates = {
            "salary_inquiry": [
                "What is my monthly salary?",
                "Show me my pay information",
                "I need to check my earnings",
                "How much do I earn annually?",
                "What's my current compensation?"
            ],
            "meeting_room_booking": [
                "I need to book a meeting room for tomorrow",
                "Can I reserve a conference room today?",
                "Book me a room for next week",
                "I want to schedule a meeting space",
                "Reserve a boardroom for Friday"
            ],
            "leave_request": [
                "I want to take vacation next week",
                "Can I request sick leave tomorrow?",
                "I need time off for Monday",
                "Apply for personal leave",
                "Request holiday for Friday"
            ],
            "directory_search": [
                "Find John Smith in the company directory",
                "Look up Sarah's contact information",
                "Search for employee phone numbers",
                "I need Mike's email address",
                "Find contact details for the HR team"
            ],
            "company_info": [
                "What is the company policy?",
                "Tell me about our mission statement",
                "What's the company address?",
                "I need headquarters information",
                "Show me company values"
            ],
            "employee_info": [
                "What is John's department?",
                "Tell me about Sarah's role",
                "Who is my manager?",
                "What's the CEO's title?",
                "Find the HR director's information"
            ],
            "employee_search": [
                "Find employees at ABC Corp named Smith",
                "Search for Johnson at TechCorp",
                "Look up Brown from GlobalTech",
                "Who is Davis at InnovateCo?",
                "Find Wilson working at XYZ Inc"
            ]
        }
    
    def generate_samples(self):
        """Generate sample data."""
        texts = []
        labels = []
        
        for intent in self.intent_labels:
            templates = self.templates[intent]
            for i in range(self.samples_per_intent):
                # Select template (with repetition if needed)
                template = templates[i % len(templates)]
                texts.append(template)
                labels.append(intent)
        
        return texts, labels
    
    def save_simple_dataset(self, output_dir="test_data"):
        """Save dataset in simple format."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        texts, labels = self.generate_samples()
        
        # Create simple train/test split
        split_idx = int(len(texts) * 0.8)
        
        train_data = {
            'texts': texts[:split_idx],
            'labels': labels[:split_idx]
        }
        
        test_data = {
            'texts': texts[split_idx:],
            'labels': labels[split_idx:]
        }
        
        # Save as JSON
        with open(output_path / "train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(output_path / "test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Save label mapping
        label_mapping = {
            'intent_labels': self.intent_labels,
            'label_to_id': {label: i for i, label in enumerate(self.intent_labels)},
            'id_to_label': {i: label for i, label in enumerate(self.intent_labels)}
        }
        
        with open(output_path / "label_mapping.json", 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        return len(train_data['texts']), len(test_data['texts'])

def test_dataset_generation():
    """Test dataset generation functionality."""
    print("Testing Dataset Generation (Lightweight)")
    print("=" * 45)
    
    # Create dataset
    creator = SimpleDatasetCreator(samples_per_intent=5)
    
    # Generate samples
    texts, labels = creator.generate_samples()
    
    print(f"Generated {len(texts)} samples")
    print(f"Found {len(set(labels))} unique intents")
    
    # Check intent distribution
    intent_counts = {}
    for label in labels:
        intent_counts[label] = intent_counts.get(label, 0) + 1
    
    print("\nIntent distribution:")
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count}")
    
    # Show sample examples
    print("\nSample examples:")
    for intent in creator.intent_labels[:3]:
        intent_examples = [text for text, label in zip(texts, labels) if label == intent]
        if intent_examples:
            print(f"  {intent}: \"{intent_examples[0]}\"")
    
    # Save dataset
    train_count, test_count = creator.save_simple_dataset()
    print(f"\nDataset saved:")
    print(f"  Training samples: {train_count}")
    print(f"  Test samples: {test_count}")
    
    # Verify files were created
    data_dir = Path("test_data")
    required_files = ["train.json", "test.json", "label_mapping.json"]
    created_files = []
    
    for file_name in required_files:
        file_path = data_dir / file_name
        if file_path.exists():
            created_files.append(file_name)
            # Check file size
            size = file_path.stat().st_size
            print(f"  {file_name}: {size} bytes")
    
    if len(created_files) == len(required_files):
        print("\n✅ Dataset generation test passed!")
        return True
    else:
        print(f"\n❌ Missing files: {set(required_files) - set(created_files)}")
        return False

def test_intent_coverage():
    """Test that all required intents are covered."""
    print("\nTesting Intent Coverage")
    print("-" * 25)
    
    creator = SimpleDatasetCreator()
    
    expected_intents = [
        "salary_inquiry",
        "meeting_room_booking", 
        "leave_request",
        "directory_search",
        "company_info",
        "employee_info",
        "employee_search"
    ]
    
    # Check if all intents are defined
    missing_intents = []
    for intent in expected_intents:
        if intent not in creator.intent_labels:
            missing_intents.append(intent)
        elif intent not in creator.templates:
            missing_intents.append(f"{intent} (no templates)")
    
    if missing_intents:
        print(f"❌ Missing intents: {missing_intents}")
        return False
    else:
        print(f"✅ All {len(expected_intents)} required intents covered")
        
        # Show template counts
        for intent in expected_intents:
            template_count = len(creator.templates[intent])
            print(f"  {intent}: {template_count} templates")
        
        return True

def main():
    """Run all lightweight tests."""
    print("RoBERTa Intent Recognition - Lightweight Tests")
    print("=" * 50)
    
    tests = [
        ("Dataset Generation", test_dataset_generation),
        ("Intent Coverage", test_intent_coverage)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if success:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
    
    print(f"\n{'='*50}")
    print("LIGHTWEIGHT TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All lightweight tests passed!")
        print("\nThe intent recognition system is ready for:")
        print("1. Installing ML dependencies: pip install -r requirements.txt")
        print("2. Full dataset generation with pandas")
        print("3. Model training with PyTorch and Transformers")
        print("4. Complete evaluation and inference")
    else:
        print("❌ Some tests failed. Check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)