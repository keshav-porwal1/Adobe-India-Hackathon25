# Adobe Hackathon Evaluation Pipeline

This evaluation pipeline provides comprehensive tools to assess the accuracy and precision of both Round 1A and Round 1B implementations.

## 📁 Project Structure

```
Adobe_Hackathon/
├── input_1A/                    # Multilingual PDFs for Round 1A
├── output_1A/                   # JSON outputs from Round 1A
├── input_1B/                    # Research PDFs for Round 1B
├── output_1B/                   # JSON outputs from Round 1B
├── ground_truth/                # Manual annotations (created by you)
├── evaluate_accuracy.py         # Main evaluation script
├── validate_multilingual.py     # Multilingual text validation
├── create_ground_truth.py       # Ground truth template creator
├── run_evaluation.py           # Complete pipeline runner
└── EVALUATION_README.md        # This file
```

## 🎯 Evaluation Objectives

### Round 1A: Heading Detection
- **Task**: Detect titles and hierarchical headings (H1, H2, H3) from multilingual PDFs
- **Metrics**: Precision, Recall, F1 Score
- **Multilingual Check**: Devanagari text correctness

### Round 1B: Document Intelligence
- **Task**: Extract relevant sections based on persona and job description
- **Metrics**: Semantic relevance, ranking accuracy, bilingual preservation
- **Output**: Ranked sections with refined text

## 🚀 Quick Start

### 1. Run Complete Evaluation Pipeline
```bash
python run_evaluation.py --mode both
```

### 2. Run Individual Evaluations
```bash
# Round 1A only
python run_evaluation.py --mode r1a

# Round 1B only
python run_evaluation.py --mode r1b
```

### 3. Create Ground Truth Templates
```bash
python run_evaluation.py --create-ground-truth
```

## 📊 Evaluation Components

### 1. Multilingual Text Validation
```bash
python validate_multilingual.py --input_dir output_1A --detailed
```

**What it checks:**
- Valid Devanagari Unicode characters
- Mixed script detection
- Garbled text patterns
- Unicode encoding issues

**Output:**
- `multilingual_validation_report.json` with detailed analysis

### 2. Accuracy Evaluation
```bash
python evaluate_accuracy.py --mode r1a --detailed
python evaluate_accuracy.py --mode r1b --detailed
```

**R1A Metrics:**
- Heading detection precision/recall/F1
- Title detection accuracy
- Multilingual text correctness
- Language detection accuracy

**R1B Metrics:**
- Semantic relevance scoring
- Keyword matching accuracy
- Bilingual content preservation
- Ranking quality assessment

### 3. Ground Truth Creation
```bash
python create_ground_truth.py --mode both
```

**Creates templates for:**
- Manual heading annotations
- Expected section rankings
- Multilingual correctness verification

## 📋 Evaluation Results

### Current Status (Based on your outputs):

#### Round 1A Results:
- **EHMMQA_English_Hindi_and_Marathi_multilingual_ques.json**:
  - ✅ 429 headings detected
  - ✅ Title detected (English)
  - ⚠️ 229 multilingual issues found (46.7% accuracy)
  - Issues: Mixed text fragments, incomplete headings

- **अबोली.json**:
  - ❌ No headings detected
  - ❌ No title detected
  - ⚠️ Empty output - needs investigation

#### Round 1B Results:
- **r1b_output.json**:
  - ✅ 165 sections extracted
  - ❌ 0 keyword matches (0% accuracy)
  - ❌ 0 bilingual sections (0% bilingual preservation)
  - ⚠️ All sections are English-only

## 🔧 Troubleshooting

### Common Issues:

1. **Empty R1A Outputs**
   ```bash
   # Check if PDFs are readable
   python r1a_outline_extractor.py
   ```

2. **Multilingual Issues**
   ```bash
   # Validate text encoding
   python validate_multilingual.py --input_dir output_1A --detailed
   ```

3. **Poor R1B Relevance**
   ```bash
   # Check persona and job description
   cat input_1B/persona.txt
   cat input_1B/job.txt
   ```

### Fixing Issues:

1. **For garbled Devanagari text:**
   - Check PDF encoding
   - Verify font embedding
   - Use OCR fallback if needed

2. **For missing headings:**
   - Check PDF structure
   - Verify font size detection
   - Review heading criteria

3. **For poor semantic relevance:**
   - Update persona/job description
   - Check model loading
   - Verify input PDFs contain relevant content

## 📈 Metrics Explanation

### Precision, Recall, F1 Score
- **Precision**: Correctly identified headings / Total predicted headings
- **Recall**: Correctly identified headings / Actual correct headings  
- **F1 Score**: Harmonic mean of precision and recall

### Multilingual Accuracy
- **Valid texts**: Properly encoded Devanagari text
- **Garbled texts**: Mixed scripts or encoding issues
- **Accuracy**: Valid texts / Total texts

### Semantic Relevance
- **Keyword matches**: Sections containing relevant terms
- **Bilingual preservation**: Sections with both English and Devanagari
- **Ranking quality**: Top sections match expected relevance

## 🎯 Recommendations

### Immediate Actions:
1. **Fix अबोली.pdf processing** - Investigate why no headings are detected
2. **Improve multilingual text extraction** - Address 46.7% accuracy in R1A
3. **Enhance semantic relevance** - Improve 0% keyword matching in R1B
4. **Add bilingual content** - Ensure Devanagari text is preserved

### Long-term Improvements:
1. **Create ground truth annotations** for proper evaluation
2. **Implement OCR fallback** for problematic PDFs
3. **Fine-tune heading detection** parameters
4. **Optimize semantic model** for better relevance scoring

## 📝 Creating Ground Truth

1. **Generate templates:**
   ```bash
   python create_ground_truth.py --mode both
   ```

2. **Manually annotate:**
   - Edit files in `ground_truth/` directory
   - Mark correct/incorrect headings
   - Add multilingual text corrections
   - Specify expected section rankings

3. **Run evaluation with ground truth:**
   ```bash
   python evaluate_accuracy.py --ground_truth_dir ground_truth --detailed
   ```

## 🔍 Detailed Analysis

### Files with Issues:
- `अबोली.json`: Empty output - needs investigation
- `EHMMQA_English_Hindi_and_Marathi_multilingual_ques.json`: 229 multilingual issues
- `r1b_output.json`: Poor semantic relevance (0% keyword matches)

### Success Indicators:
- ✅ No Unicode errors in validation
- ✅ Basic heading detection working
- ✅ JSON output format correct
- ✅ Multilingual model loading successfully

## 📞 Support

For issues with the evaluation pipeline:
1. Check the detailed error messages
2. Verify all dependencies are installed
3. Ensure input files are in correct format
4. Review the troubleshooting section above

---

**Last Updated**: 2025-01-26
**Pipeline Version**: 1.0 