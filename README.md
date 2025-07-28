# Adobe Challenge 1A - PDF Heading Extraction System

A high-accuracy PDF parsing system for extracting titles and hierarchical headings (H1, H2, H3) from multilingual PDFs.

## 🎯 Features

- **Title Extraction**: Automatically detects document titles from the first page
- **Hierarchical Heading Detection**: Identifies H1, H2, and H3 headings based on font properties
- **Multilingual Support**: Handles documents with mixed scripts (English, Devanagari, etc.)
- **Noise Filtering**: Removes headers, footers, and other non-content elements
- **Font Clustering**: Uses machine learning to dynamically classify heading tiers
- **Docker Ready**: Containerized solution for easy deployment

## 🚀 Quick Start

### 1. Local Development

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Test with sample dataset
python test_extraction.py
```

### 2. Docker Deployment

```bash
# Build Docker image
docker build -t adobe-challenge-1a .

# Run with mounted volumes
docker run -v /path/to/input:/app/input -v /path/to/output:/app/output adobe-challenge-1a
```

## 📁 Project Structure

```
Adobe_1A/
├── process_pdfs.py          # Main extraction script
├── test_extraction.py       # Test script for sample dataset
├── requirements.txt         # Python dependencies
├── dockerfile              # Docker configuration
├── README.md               # This file
└── challange_1A/
    └── sample_dataset/
        ├── pdfs/           # Input PDF files
        └── outputs/        # Generated JSON outputs
```

## 🔧 How It Works

### 1. Font Analysis
- Extracts all text spans with font properties (size, boldness, position)
- Uses K-means clustering to identify font size tiers
- Calculates median body text size for relative comparison

### 2. Title Detection
- Analyzes first page for largest font sizes
- Prefers bold text for title candidates
- Concatenates multiple title lines in correct order
- Filters out common non-title patterns

### 3. Heading Detection
- **H1**: Font size ≥ 1.4× median, bold, ≥14pt
- **H2**: Font size ≥ 1.2× median, bold, ≥12pt  
- **H3**: Font size ≥ 1.1× median, (bold or ≥11pt)

### 4. Post-Processing
- Merges split headings across multiple spans
- Removes duplicates and noise
- Ensures proper hierarchical structure
- Filters out headers/footers

## 📊 Output Format

```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Main Heading",
      "page": 1
    },
    {
      "level": "H2", 
      "text": "Subheading",
      "page": 2
    }
  ]
}
```

## 🧪 Testing

### Sample Dataset Results

| File | Title | Headings | H1 | H2 | H3 |
|------|-------|----------|----|----|----|
| file01.pdf | Application form... | 0 | 0 | 0 | 0 |
| file02.pdf | Overview Foundation... | 7 | 7 | 0 | 0 |
| file03.pdf | RFP: Request for... | 23 | 2 | 10 | 11 |
| file04.pdf | Parsippany STEM... | 2 | 1 | 1 | 0 |
| file05.pdf | HOPE... | 1 | 1 | 0 | 0 |

## ⚙️ Configuration

### Key Parameters

```python
class PDFHeadingExtractor:
    def __init__(self):
        self.min_heading_length = 3      # Minimum heading length
        self.max_heading_length = 200    # Maximum heading length
        self.min_font_size = 8          # Minimum font size to consider
        self.max_font_size = 72         # Maximum font size to consider
```

### Font Detection Criteria

- **Bold Detection**: Checks font name and PyMuPDF flags
- **Size Thresholds**: Relative to median body text size
- **Clustering**: Uses K-means with 3 clusters max

## 🔍 Performance

- **Speed**: < 10 seconds per 50-page PDF
- **Memory**: < 200MB model size
- **Accuracy**: High precision for well-structured documents
- **Robustness**: Handles various PDF formats and layouts

## 🛠️ Dependencies

- **PyMuPDF**: PDF text extraction and font analysis
- **scikit-learn**: Font size clustering
- **numpy**: Numerical operations
- **pandas**: Data manipulation (optional)

## 🐳 Docker Constraints

- **Architecture**: AMD64 compatible
- **Memory**: 16GB RAM limit
- **CPU**: 8-core limit
- **No Internet**: Offline operation required
- **Input/Output**: `/app/input` and `/app/output` directories

## 🎯 Use Cases

1. **Document Processing**: Extract structure from technical documents
2. **Content Analysis**: Identify document sections and hierarchy
3. **Multilingual Support**: Handle documents with mixed scripts
4. **Automation**: Integrate with document management systems

## 🔧 Troubleshooting

### Common Issues

1. **No Headings Detected**
   - Check PDF structure and font embedding
   - Verify font size thresholds
   - Review heading detection criteria

2. **Poor Title Extraction**
   - Adjust title tolerance (currently 0.85)
   - Check for bold text preference
   - Review noise filtering patterns

3. **Performance Issues**
   - Reduce clustering complexity
   - Optimize font size thresholds
   - Consider document preprocessing

## 📈 Future Improvements

- **OCR Fallback**: For problematic PDFs
- **Layout Analysis**: Better column and section detection
- **Machine Learning**: Train on annotated datasets
- **Multilingual Enhancement**: Better script detection
- **Validation**: Schema-based output validation

## 📄 License

This project is developed for Adobe Challenge 1A.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with sample dataset
5. Submit a pull request

---

**Note**: This system is optimized for Adobe Challenge 1A requirements and constraints. 