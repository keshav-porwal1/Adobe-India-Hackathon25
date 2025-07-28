#!/usr/bin/env python3
"""
Complete test script for Adobe Challenge 1A PDF extraction system
Runs the entire pipeline and provides comprehensive status report
"""

import os
import json
import time
from test_extraction import test_with_sample_dataset
from validate_outputs import validate_outputs, print_validation_report

def run_complete_test():
    """Run the complete test pipeline."""
    print("=" * 70)
    print("ADOBE CHALLENGE 1A - COMPLETE SYSTEM TEST")
    print("=" * 70)
    
    # Step 1: Test PDF extraction
    print("\n🔍 STEP 1: PDF Extraction Test")
    print("-" * 40)
    
    start_time = time.time()
    extraction_results = test_with_sample_dataset()
    extraction_time = time.time() - start_time
    
    print(f"\n✅ Extraction completed in {extraction_time:.2f} seconds")
    
    # Step 2: Validate outputs
    print("\n🔍 STEP 2: Output Validation")
    print("-" * 40)
    
    output_dir = "challange_1A/sample_dataset/outputs"
    validation_results = validate_outputs(output_dir)
    print_validation_report(validation_results)
    
    # Step 3: Generate summary report
    print("\n🔍 STEP 3: System Summary")
    print("-" * 40)
    
    total_headings = 0
    total_h1 = 0
    total_h2 = 0
    total_h3 = 0
    
    for filename, result in validation_results['files'].items():
        if result['valid']:
            total_headings += result['headings_count']
            total_h1 += result['h1_count']
            total_h2 += result['h2_count']
            total_h3 += result['h3_count']
    
    print(f"📊 EXTRACTION STATISTICS:")
    print(f"   Files processed: {validation_results['total_files']}")
    print(f"   Total headings extracted: {total_headings}")
    print(f"   H1 headings: {total_h1}")
    print(f"   H2 headings: {total_h2}")
    print(f"   H3 headings: {total_h3}")
    print(f"   Average headings per file: {total_headings/validation_results['total_files']:.1f}")
    
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"   Processing time: {extraction_time:.2f} seconds")
    print(f"   Average time per file: {extraction_time/validation_results['total_files']:.2f} seconds")
    print(f"   Success rate: {validation_results['valid_files']/validation_results['total_files']*100:.1f}%")
    
    print(f"\n🎯 SYSTEM STATUS:")
    if validation_results['valid_files'] == validation_results['total_files']:
        print("   ✅ All files processed successfully")
        print("   ✅ All outputs are valid JSON")
        print("   ✅ Ready for Adobe Challenge 1A")
    else:
        print("   ❌ Some files failed processing")
        print("   ⚠️  Check error logs for details")
    
    # Step 4: Check Docker readiness
    print(f"\n🐳 DOCKER READINESS:")
    docker_files = ['process_pdfs.py', 'requirements.txt', 'dockerfile']
    missing_files = [f for f in docker_files if not os.path.exists(f)]
    
    if not missing_files:
        print("   ✅ All required files present")
        print("   ✅ Ready for Docker build")
    else:
        print(f"   ❌ Missing files: {missing_files}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)
    
    return {
        'extraction_results': extraction_results,
        'validation_results': validation_results,
        'performance': {
            'total_time': extraction_time,
            'files_processed': validation_results['total_files'],
            'success_rate': validation_results['valid_files']/validation_results['total_files']*100
        }
    }

if __name__ == "__main__":
    run_complete_test() 