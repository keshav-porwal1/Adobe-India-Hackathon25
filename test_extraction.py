#!/usr/bin/env python3
"""
Test script for PDF heading extraction using sample dataset
"""

import os
import json
import sys
from process_pdfs import PDFHeadingExtractor

def test_with_sample_dataset():
    """Test the extraction with sample PDFs."""
    sample_pdfs_dir = "challange_1A/sample_dataset/pdfs"
    output_dir = "challange_1A/sample_dataset/outputs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize extractor
    extractor = PDFHeadingExtractor()
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(sample_pdfs_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in sample dataset")
        return
    
    print(f"Found {len(pdf_files)} PDF files to test")
    
    results = {}
    
    for filename in pdf_files:
        pdf_path = os.path.join(sample_pdfs_dir, filename)
        output_filename = filename.replace('.pdf', '.json')
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\nProcessing {filename}...")
        
        try:
            result = extractor.extract_document_info(pdf_path)
            
            # Write output
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # Store results for summary
            results[filename] = {
                'title': result['title'],
                'headings_count': len(result['outline']),
                'h1_count': len([h for h in result['outline'] if h['level'] == 'H1']),
                'h2_count': len([h for h in result['outline'] if h['level'] == 'H2']),
                'h3_count': len([h for h in result['outline'] if h['level'] == 'H3']),
                'output_path': output_path
            }
            
            print(f"✅ Generated: {output_path}")
            print(f"   Title: {result['title'][:50]}...")
            print(f"   Headings: {len(result['outline'])} (H1: {results[filename]['h1_count']}, H2: {results[filename]['h2_count']}, H3: {results[filename]['h3_count']})")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            results[filename] = {'error': str(e)}
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    for filename, result in results.items():
        if 'error' in result:
            print(f"❌ {filename}: {result['error']}")
        else:
            print(f"✅ {filename}:")
            print(f"   Title: {result['title'][:30]}...")
            print(f"   Headings: {result['headings_count']} (H1: {result['h1_count']}, H2: {result['h2_count']}, H3: {result['h3_count']})")
    
    return results

if __name__ == "__main__":
    test_with_sample_dataset() 