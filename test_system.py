"""
Simple test script to verify the intent recognition system structure and basic functionality.

This script tests the modules without requiring heavy ML dependencies,
focusing on imports, data structures, and basic logic.
"""

import os
import sys
from pathlib import Path

# Add the python_intent_recognition directory to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_project_structure():
    """Test that all required files and directories exist."""
    print("Testing project structure...")
    
    required_files = [
        "README.md",
        "requirements.txt",
        "__init__.py",
        "dataset/__init__.py",
        "dataset/dataset_creator.py",
        "models/__init__.py", 
        "models/roberta_classifier.py",
        "training/__init__.py",
        "training/train.py",
        "evaluation/__init__.py",
        "evaluation/evaluate.py",
        "inference/__init__.py",
        "inference/predict.py",
        "examples/__init__.py",
        "examples/demo.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files present")
        return True

def test_basic_imports():
    """Test basic imports without heavy dependencies."""
    print("\nTesting basic imports...")
    
    try:
        # Test intent labels by importing from __init__.py
        init_file = project_root / "__init__.py"
        if init_file.exists():
            # Read and check content
            with open(init_file, 'r') as f:
                content = f.read()
            
            if 'INTENT_LABELS' in content:
                print("✓ Intent labels defined in __init__.py")
                
                # Extract intent labels count
                import re
                labels_match = re.search(r'INTENT_LABELS\s*=\s*\[(.*?)\]', content, re.DOTALL)
                if labels_match:
                    labels_content = labels_match.group(1)
                    # Count non-empty lines with quotes
                    intent_count = len([line.strip() for line in labels_content.split(',') if line.strip() and '"' in line])
                    print(f"  Found {intent_count} intents defined")
                
                if 'LABEL_TO_ID' in content and 'ID_TO_LABEL' in content:
                    print("✓ Label mappings defined")
                else:
                    print("✗ Label mappings missing")
                    return False
            else:
                print("✗ Intent labels not found in __init__.py")
                return False
        else:
            print("✗ __init__.py file not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_dataset_templates():
    """Test dataset generation logic without pandas dependency."""
    print("\nTesting dataset generation templates...")
    
    try:
        # Read the dataset creator file and check for required components
        dataset_file = project_root / "dataset" / "dataset_creator.py"
        with open(dataset_file, 'r') as f:
            content = f.read()
        
        # Check for key components
        required_components = [
            "OfficeIntentDatasetCreator",
            "salary_templates",
            "meeting_templates", 
            "leave_templates",
            "directory_templates",
            "company_templates",
            "employee_templates",
            "search_templates"
        ]
        
        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            print(f"✗ Missing components: {missing_components}")
            return False
        else:
            print("✓ All dataset templates present")
            return True
            
    except Exception as e:
        print(f"✗ Error checking dataset templates: {e}")
        return False

def test_requirements():
    """Test that requirements.txt has necessary dependencies."""
    print("\nTesting requirements...")
    
    try:
        req_file = project_root / "requirements.txt"
        with open(req_file, 'r') as f:
            requirements = f.read()
        
        required_deps = ['torch', 'transformers', 'pandas', 'scikit-learn', 'numpy']
        missing_deps = []
        
        for dep in required_deps:
            if dep not in requirements:
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"✗ Missing dependencies: {missing_deps}")
            return False
        else:
            print("✓ All required dependencies listed")
            return True
            
    except Exception as e:
        print(f"✗ Error checking requirements: {e}")
        return False

def test_documentation():
    """Test that README and documentation are present."""
    print("\nTesting documentation...")
    
    try:
        readme_file = project_root / "README.md"
        with open(readme_file, 'r') as f:
            readme_content = f.read()
        
        required_sections = [
            "Supported Intents",
            "Project Structure", 
            "Quick Start",
            "Implementation Details"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in readme_content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"✗ Missing README sections: {missing_sections}")
            return False
        else:
            print("✓ README documentation complete")
            return True
            
    except Exception as e:
        print(f"✗ Error checking documentation: {e}")
        return False

def main():
    """Run all tests."""
    print("RoBERTa Intent Recognition System - Structure Test")
    print("=" * 55)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Basic Imports", test_basic_imports),
        ("Dataset Templates", test_dataset_templates),
        ("Requirements", test_requirements),
        ("Documentation", test_documentation)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if success:
                passed_tests += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    print(f"\n{'='*55}")
    print("TEST SUMMARY")
    print(f"{'='*55}")
    print(f"Passed: {passed_tests}/{total_tests} tests")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The intent recognition system is properly structured.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Generate dataset: python -m dataset.dataset_creator")
        print("3. Train model: python -m training.train")
        print("4. Run demo: python -m examples.demo --quick")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    main()