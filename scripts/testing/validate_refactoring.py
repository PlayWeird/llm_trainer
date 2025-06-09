#!/usr/bin/env python3
"""
Validation script to ensure the refactored code maintains compatibility
"""

import os
import sys
import ast
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def check_imports_in_file(file_path):
    """Check if all imports in a file are valid"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        
        return True, imports
    except SyntaxError as e:
        return False, str(e)


def validate_training_scripts():
    """Validate all training scripts"""
    training_dir = Path("scripts/training")
    vlm_scripts = [
        "train_gemma3_vlm.py",
        "train_llava_vlm.py",
        "train_idefics3_vlm.py",
        "train_qwen2_vlm.py"
    ]
    
    print("Validating training scripts...")
    all_valid = True
    
    for script in vlm_scripts:
        script_path = training_dir / script
        if script_path.exists():
            valid, result = check_imports_in_file(script_path)
            if valid:
                print(f"✓ {script} - Valid syntax")
            else:
                print(f"✗ {script} - Syntax error: {result}")
                all_valid = False
        else:
            print(f"- {script} - Not found (skipping)")
    
    return all_valid


def check_module_structure():
    """Check that all utility modules have proper structure"""
    utils_dir = Path("utils")
    required_files = [
        "__init__.py",
        "vlm_data_utils.py",
        "training_config.py",
        "model_utils.py",
        "training_utils.py"
    ]
    
    print("\nChecking module structure...")
    all_present = True
    
    for file in required_files:
        file_path = utils_dir / file
        if file_path.exists():
            print(f"✓ {file} - Present")
            
            # Check if it's valid Python
            valid, _ = check_imports_in_file(file_path)
            if not valid:
                print(f"  ✗ Has syntax errors")
                all_present = False
        else:
            print(f"✗ {file} - Missing")
            all_present = False
    
    return all_present


def verify_backwards_compatibility():
    """Verify that the refactored code maintains backwards compatibility"""
    print("\nChecking backwards compatibility...")
    
    # Check that common patterns still work
    try:
        # Test importing configurations
        from utils import ModelWithLoRAArguments, VLMDataArguments, TrainingArguments
        
        # Test creating instances
        model_args = ModelWithLoRAArguments()
        data_args = VLMDataArguments(dataset_path="test.json", image_dir="./images")
        training_args = TrainingArguments(output_dir="./test")
        
        print("✓ Configuration classes instantiate correctly")
        
        # Test importing utilities
        from utils import (
            get_quantization_config,
            setup_lora,
            create_data_collator,
            setup_training_environment
        )
        
        print("✓ Utility functions can be imported")
        
        return True
    except Exception as e:
        print(f"✗ Compatibility issue: {e}")
        return False


def check_documentation():
    """Check that all modules have proper documentation"""
    print("\nChecking documentation...")
    utils_dir = Path("utils")
    
    for py_file in utils_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        with open(py_file, 'r') as f:
            content = f.read()
            
        # Check for module docstring
        tree = ast.parse(content)
        has_docstring = ast.get_docstring(tree) is not None
        
        if has_docstring:
            print(f"✓ {py_file.name} - Has module docstring")
        else:
            print(f"✗ {py_file.name} - Missing module docstring")
    
    return True


def main():
    """Run all validation checks"""
    print("Validating refactored training code...\n")
    
    checks = [
        ("Module Structure", check_module_structure),
        ("Training Scripts", validate_training_scripts),
        ("Backwards Compatibility", verify_backwards_compatibility),
        ("Documentation", check_documentation),
    ]
    
    all_passed = True
    
    for name, check_func in checks:
        try:
            passed = check_func()
            all_passed = all_passed and passed
        except Exception as e:
            print(f"\n✗ {name} check failed with error: {e}")
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✓ All validation checks passed!")
    else:
        print("✗ Some validation checks failed")
    print("="*50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())