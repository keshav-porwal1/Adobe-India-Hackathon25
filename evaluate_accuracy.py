#!/usr/bin/env python3
"""
Accuracy Evaluation Script for Adobe Challenge 1A
Compares generated outputs with original expected outputs
"""

import json
import os
import re
from typing import Dict, List, Tuple, Any
from difflib import SequenceMatcher

def clean_text_for_comparison(text: str) -> str:
    """Clean text for comparison by removing extra spaces and normalizing."""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Remove common variations
    text = text.replace('  ', ' ')  # Double spaces
    text = text.replace(' ,', ',')  # Space before comma
    text = text.replace(' .', '.')  # Space before period
    
    return text.lower()

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using SequenceMatcher."""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    
    clean1 = clean_text_for_comparison(text1)
    clean2 = clean_text_for_comparison(text2)
    
    return SequenceMatcher(None, clean1, clean2).ratio()

def compare_headings(expected_headings: List[Dict], actual_headings: List[Dict]) -> Dict[str, Any]:
    """Compare expected and actual headings."""
    results = {
        'total_expected': len(expected_headings),
        'total_actual': len(actual_headings),
        'exact_matches': 0,
        'partial_matches': 0,
        'unmatched_expected': [],
        'unmatched_actual': [],
        'matched_pairs': []
    }
    
    # Create copies for matching
    expected_remaining = expected_headings.copy()
    actual_remaining = actual_headings.copy()
    
    # First pass: find exact matches
    for exp_heading in expected_headings[:]:
        for act_heading in actual_headings[:]:
            if (exp_heading.get('level') == act_heading.get('level') and
                clean_text_for_comparison(exp_heading.get('text', '')) == clean_text_for_comparison(act_heading.get('text', ''))):
                
                results['exact_matches'] += 1
                results['matched_pairs'].append({
                    'expected': exp_heading,
                    'actual': act_heading,
                    'match_type': 'exact',
                    'similarity': 1.0
                })
                
                if exp_heading in expected_remaining:
                    expected_remaining.remove(exp_heading)
                if act_heading in actual_remaining:
                    actual_remaining.remove(act_heading)
                break
    
    # Second pass: find partial matches
    for exp_heading in expected_remaining[:]:
        best_match = None
        best_similarity = 0.0
        
        for act_heading in actual_remaining[:]:
            similarity = calculate_text_similarity(
                exp_heading.get('text', ''),
                act_heading.get('text', '')
            )
            
            if similarity > 0.7 and similarity > best_similarity:  # Threshold for partial match
                best_similarity = similarity
                best_match = act_heading
        
        if best_match:
            results['partial_matches'] += 1
            results['matched_pairs'].append({
                'expected': exp_heading,
                'actual': best_match,
                'match_type': 'partial',
                'similarity': best_similarity
            })
            
            expected_remaining.remove(exp_heading)
            actual_remaining.remove(best_match)
    
    # Remaining unmatched
    results['unmatched_expected'] = expected_remaining
    results['unmatched_actual'] = actual_remaining
    
    return results

def evaluate_file(org_output_path: str, actual_output_path: str) -> Dict[str, Any]:
    """Evaluate accuracy for a single file."""
    try:
        # Load original expected output
        with open(org_output_path, 'r', encoding='utf-8') as f:
            expected_data = json.load(f)
        
        # Load actual output
        with open(actual_output_path, 'r', encoding='utf-8') as f:
            actual_data = json.load(f)
        
        # Compare titles
        expected_title = expected_data.get('title', '')
        actual_title = actual_data.get('title', '')
        
        title_similarity = calculate_text_similarity(expected_title, actual_title)
        title_match = title_similarity > 0.8  # Threshold for title match
        
        # Compare headings
        expected_headings = expected_data.get('outline', [])
        actual_headings = actual_data.get('outline', [])
        
        heading_results = compare_headings(expected_headings, actual_headings)
        
        # Calculate accuracy metrics
        total_expected_headings = len(expected_headings)
        total_actual_headings = len(actual_headings)
        total_matches = heading_results['exact_matches'] + heading_results['partial_matches']
        
        heading_accuracy = total_matches / total_expected_headings if total_expected_headings > 0 else 0.0
        
        return {
            'file_name': os.path.basename(org_output_path),
            'title': {
                'expected': expected_title,
                'actual': actual_title,
                'match': title_match,
                'similarity': title_similarity
            },
            'headings': {
                'expected_count': total_expected_headings,
                'actual_count': total_actual_headings,
                'exact_matches': heading_results['exact_matches'],
                'partial_matches': heading_results['partial_matches'],
                'total_matches': total_matches,
                'accuracy': heading_accuracy,
                'details': heading_results
            },
            'overall_accuracy': (title_similarity + heading_accuracy) / 2
        }
        
    except Exception as e:
        return {
            'file_name': os.path.basename(org_output_path),
            'error': str(e),
            'title': {'match': False, 'similarity': 0.0},
            'headings': {'accuracy': 0.0, 'total_matches': 0},
            'overall_accuracy': 0.0
        }

def print_file_report(file_result: Dict[str, Any]):
    """Print detailed report for a single file."""
    filename = file_result['file_name'].upper().replace('.JSON', '')
    print(f"\n{filename} ===\n")
    
    if 'error' in file_result:
        print(f"Error: {file_result['error']}")
        return
    
    # Title comparison
    title_info = file_result['title']
    print(f"Title Match: {title_info['match']}")
    print(f"Expected: \"{title_info['expected']}\"")
    print(f"Actual: \"{title_info['actual']}\"")
    
    # Heading comparison
    heading_info = file_result['headings']
    print(f"\nExpected headings: {heading_info['expected_count']}")
    print(f"Actual headings: {heading_info['actual_count']}")
    print(f"Exact matches: {heading_info['exact_matches']}/{heading_info['expected_count']} ({heading_info['accuracy']*100:.1f}%)")
    
    # Show unmatched headings if any
    details = heading_info['details']
    if details['unmatched_expected']:
        print(f"\nUnmatched expected headings:")
        for heading in details['unmatched_expected']:
            print(f"  - {heading.get('text', '')} ({heading.get('level', '')})")
    
    if details['unmatched_actual']:
        print(f"\nUnmatched actual headings:")
        for heading in details['unmatched_actual']:
            print(f"  - {heading.get('text', '')} ({heading.get('level', '')})")

def evaluate_all_files():
    """Evaluate accuracy for all files."""
    org_outputs_dir = "challange_1A/sample_dataset/org_outputs"
    actual_outputs_dir = "challange_1A/sample_dataset/outputs"
    
    if not os.path.exists(org_outputs_dir):
        print(f"Original outputs directory not found: {org_outputs_dir}")
        return
    
    if not os.path.exists(actual_outputs_dir):
        print(f"Actual outputs directory not found: {actual_outputs_dir}")
        return
    
    # Get all JSON files
    org_files = [f for f in os.listdir(org_outputs_dir) if f.endswith('.json')]
    actual_files = [f for f in os.listdir(actual_outputs_dir) if f.endswith('.json')]
    
    print("=" * 60)
    print("ADOBE CHALLENGE 1A - ACCURACY EVALUATION")
    print("=" * 60)
    
    all_results = []
    total_title_matches = 0
    total_expected_headings = 0
    total_matched_headings = 0
    
    for org_file in org_files:
        if org_file in actual_files:
            org_path = os.path.join(org_outputs_dir, org_file)
            actual_path = os.path.join(actual_outputs_dir, org_file)
            
            result = evaluate_file(org_path, actual_path)
            all_results.append(result)
            
            # Print individual file report
            print_file_report(result)
            
            # Accumulate totals
            if 'error' not in result:
                if result['title']['match']:
                    total_title_matches += 1
                total_expected_headings += result['headings']['expected_count']
                total_matched_headings += result['headings']['total_matches']
    
    # Print overall summary
    print("\n" + "=" * 60)
    print("OVERALL ACCURACY")
    print("=" * 60)
    
    total_files = len(all_results)
    title_accuracy = total_title_matches / total_files if total_files > 0 else 0
    heading_accuracy = total_matched_headings / total_expected_headings if total_expected_headings > 0 else 0
    combined_accuracy = (title_accuracy + heading_accuracy) / 2
    
    print(f"\nTitle accuracy: {total_title_matches}/{total_files} ({title_accuracy*100:.1f}%)")
    print(f"Total headings expected: {total_expected_headings}")
    print(f"Total headings matched: {total_matched_headings}")
    print(f"Heading accuracy: {heading_accuracy*100:.1f}%")
    print(f"Combined accuracy: {combined_accuracy*100:.1f}%")
    
    # Save detailed results to file
    detailed_results = {
        'summary': {
            'total_files': total_files,
            'title_accuracy': title_accuracy,
            'heading_accuracy': heading_accuracy,
            'combined_accuracy': combined_accuracy,
            'total_expected_headings': total_expected_headings,
            'total_matched_headings': total_matched_headings
        },
        'file_results': all_results
    }
    
    with open('accuracy_report.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: accuracy_report.json")

if __name__ == "__main__":
    evaluate_all_files() 