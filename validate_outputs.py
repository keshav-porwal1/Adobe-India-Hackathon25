#!/usr/bin/env python3
"""
Validation script for Adobe Challenge 1A outputs
"""

import json
import os
import sys
from typing import Dict, List, Any

def validate_json_structure(data: Dict) -> List[str]:
    """Validate JSON structure against expected schema."""
    errors = []
    
    # Check required top-level keys
    required_keys = ['title', 'outline']
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")
    
    if errors:
        return errors
    
    # Validate title
    if not isinstance(data['title'], str):
        errors.append("Title must be a string")
    
    # Validate outline
    if not isinstance(data['outline'], list):
        errors.append("Outline must be a list")
        return errors
    
    # Validate each heading in outline
    for i, heading in enumerate(data['outline']):
        if not isinstance(heading, dict):
            errors.append(f"Heading {i} must be a dictionary")
            continue
        
        # Check required heading keys
        heading_keys = ['level', 'text', 'page']
        for key in heading_keys:
            if key not in heading:
                errors.append(f"Heading {i} missing required key: {key}")
        
        # Validate heading values
        if 'level' in heading and heading['level'] not in ['H1', 'H2', 'H3']:
            errors.append(f"Heading {i} has invalid level: {heading['level']}")
        
        if 'text' in heading and not isinstance(heading['text'], str):
            errors.append(f"Heading {i} text must be a string")
        
        if 'page' in heading and not isinstance(heading['page'], int):
            errors.append(f"Heading {i} page must be an integer")
    
    return errors

def validate_multilingual_text(text: str) -> Dict[str, Any]:
    """Check for multilingual text issues."""
    issues = {
        'has_devanagari': False,
        'has_english': False,
        'mixed_scripts': False,
        'garbled_text': False
    }
    
    if not text:
        return issues
    
    # Check for Devanagari characters (Unicode range 0x0900-0x097F)
    devanagari_chars = [c for c in text if '\u0900' <= c <= '\u097F']
    issues['has_devanagari'] = len(devanagari_chars) > 0
    
    # Check for English characters
    english_chars = [c for c in text if c.isalpha() and ord(c) < 128]
    issues['has_english'] = len(english_chars) > 0
    
    # Check for mixed scripts
    issues['mixed_scripts'] = issues['has_devanagari'] and issues['has_english']
    
    # Check for garbled text (mixed characters in same word)
    words = text.split()
    for word in words:
        if len(word) > 3:
            devanagari_in_word = any('\u0900' <= c <= '\u097F' for c in word)
            english_in_word = any(c.isalpha() and ord(c) < 128 for c in word)
            if devanagari_in_word and english_in_word:
                issues['garbled_text'] = True
                break
    
    return issues

def validate_outputs(output_dir: str) -> Dict[str, Any]:
    """Validate all output files in the directory."""
    results = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'files': {}
    }
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return results
    
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    results['total_files'] = len(json_files)
    
    for filename in json_files:
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            structure_errors = validate_json_structure(data)
            
            # Validate multilingual text
            multilingual_issues = validate_multilingual_text(data.get('title', ''))
            for heading in data.get('outline', []):
                heading_issues = validate_multilingual_text(heading.get('text', ''))
                for key in multilingual_issues:
                    multilingual_issues[key] = multilingual_issues[key] or heading_issues[key]
            
            file_result = {
                'valid': len(structure_errors) == 0,
                'structure_errors': structure_errors,
                'multilingual_issues': multilingual_issues,
                'title': data.get('title', '')[:50] + '...',
                'headings_count': len(data.get('outline', [])),
                'h1_count': len([h for h in data.get('outline', []) if h.get('level') == 'H1']),
                'h2_count': len([h for h in data.get('outline', []) if h.get('level') == 'H2']),
                'h3_count': len([h for h in data.get('outline', []) if h.get('level') == 'H3'])
            }
            
            results['files'][filename] = file_result
            
            if file_result['valid']:
                results['valid_files'] += 1
            else:
                results['invalid_files'] += 1
                
        except Exception as e:
            results['files'][filename] = {
                'valid': False,
                'error': str(e),
                'structure_errors': [str(e)],
                'multilingual_issues': {}
            }
            results['invalid_files'] += 1
    
    return results

def print_validation_report(results: Dict[str, Any]):
    """Print a formatted validation report."""
    print("=" * 60)
    print("ADOBE CHALLENGE 1A - OUTPUT VALIDATION REPORT")
    print("=" * 60)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total files: {results['total_files']}")
    print(f"   Valid files: {results['valid_files']}")
    print(f"   Invalid files: {results['invalid_files']}")
    print(f"   Success rate: {results['valid_files']/results['total_files']*100:.1f}%" if results['total_files'] > 0 else "   Success rate: N/A")
    
    print(f"\n📋 DETAILED RESULTS:")
    for filename, result in results['files'].items():
        status = "✅ VALID" if result['valid'] else "❌ INVALID"
        print(f"\n{filename}: {status}")
        
        if result['valid']:
            print(f"   Title: {result['title']}")
            print(f"   Headings: {result['headings_count']} (H1: {result['h1_count']}, H2: {result['h2_count']}, H3: {result['h3_count']})")
            
            # Check multilingual issues
            issues = result['multilingual_issues']
            if any(issues.values()):
                print(f"   ⚠️  Multilingual issues:")
                if issues['has_devanagari']:
                    print(f"      - Contains Devanagari text")
                if issues['mixed_scripts']:
                    print(f"      - Mixed scripts detected")
                if issues['garbled_text']:
                    print(f"      - Garbled text detected")
        else:
            if 'error' in result:
                print(f"   Error: {result['error']}")
            if result['structure_errors']:
                print(f"   Structure errors:")
                for error in result['structure_errors']:
                    print(f"      - {error}")

def main():
    """Main validation function."""
    output_dir = "challange_1A/sample_dataset/outputs"
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    print(f"Validating outputs in: {output_dir}")
    results = validate_outputs(output_dir)
    print_validation_report(results)

if __name__ == "__main__":
    main() 