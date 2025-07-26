# Interactive PCA Scatterplot Matrix - Usage Guide

## Overview
We've successfully created interactive scatterplot matrices for both static and temporal PCA analysis with multiple export formats and enhanced visualizations.

## Files Created

### Main Scripts

1. **`interactive_scatterplot_matrix_pca.py`** - Static PCA analysis (genotype averages)
2. **`interactive_scatterplot_matrix_temporal_pca.py`** - Temporal PCA (fPCA) analysis (genotype averages)
3. **`interactive_scatterplot_matrix_individual.py`** - Individual fly version (static PCA)
4. **`interactive_scatterplot_matrix_temporal_individual.py`** - Individual fly version (temporal PCA)

### Output Files

1. **`interactive_pca_scatterplot_matrix.html`** - Interactive static PCA matrix (genotype averages)
2. **`interactive_temporal_pca_scatterplot_matrix.html`** - Interactive temporal PCA matrix (genotype averages)
3. **`interactive_pca_scatterplot_matrix_individual.html`** - Individual flies static PCA
4. **`interactive_temporal_pca_scatterplot_matrix_individual.html`** - Individual flies temporal PCA
5. **`pca_legend_combined.png`** - Combined legend (brain regions, significance, ellipses)

## Features Implemented

### Visual Improvements ✓
- **Histograms**: Solid lines with brain region color fills (alpha=0.35)
- **Scatter plots**: Reduced alpha (0.5) for better visibility when crowded
- **Component labeling**: PC1-6 for static, fPC1-6 for temporal (correct terminology)
- **Marker shapes**: Circles (both), squares (static only), triangles (temporal only)

### Export Capabilities ✓
- **HTML**: Interactive version with zoom, pan, hover tooltips
- **PNG**: Attempted automatic export (requires browser drivers)
- **Vector graphics**: Guidance provided for PDF → EPS/SVG conversion

### Data Analysis ✓
- **Static PCA**: PC1-6 components from classical PCA
- **Temporal PCA**: fPC1-6 components from functional PCA (FPCA)
- **Significance**: Both static and temporal statistical results integrated
- **Brain regions**: Color-coded with proper legend

## Usage

### Running the Scripts

```bash
# Static PCA matrix (genotype averages)
python interactive_scatterplot_matrix_pca.py

# Temporal PCA matrix (genotype averages)
python interactive_scatterplot_matrix_temporal_pca.py

# Individual flies (static PCA)
python interactive_scatterplot_matrix_individual.py

# Individual flies (temporal PCA)
python interactive_scatterplot_matrix_temporal_individual.py
```

### Export Options

#### 1. Automated (when browser drivers work)
- HTML: ✓ Always works
- PNG: Requires selenium + geckodriver/chromedriver
- SVG: Not directly supported by HoloViews

#### 2. Manual Export (recommended for presentations)
- **For PNG/JPG**:
  1. Open HTML file in browser
  2. Take high-resolution screenshot
  3. Or use browser's built-in export tools

- **For Vector Graphics (EPS/SVG)**:
  1. Open HTML file in browser
  2. Print to PDF (high quality)
  3. Convert PDF to EPS/SVG using external tools:
     - Adobe Illustrator
     - Inkscape (free)
     - Online converters

#### 3. Browser Print-to-PDF
- Open HTML file in browser
- Use Print → Save as PDF
- Select appropriate page size and quality
- Convert PDF to desired format

## Differences Between Versions

### Static vs Temporal PCA
- **Static**: Classical PCA on static metrics (PC1-6)
- **Temporal**: Functional PCA on time series data (fPC1-6)
- **Data**: Same flies, different analytical approaches

### Genotype Averages vs Individual Flies
- **Genotype averages**: Mean values per genotype (cleaner visualization)
- **Individual flies**: All data points (more detail, 831 flies total)

## Interactive Features

### Hover Tooltips
- Genotype name
- Brain region
- Significance type
- Number of flies
- PC/fPC values
- Statistical p-values

### Navigation
- **Pan**: Click and drag
- **Zoom**: Mouse wheel or box zoom
- **Reset**: Reset button in toolbar
- **Save**: Export current view

### Visual Elements
- **Control ellipses**: 95% confidence intervals for control lines
- **Reference lines**: Zero lines for orientation
- **Color coding**: Brain region colors from Config.py
- **Size/shape coding**: Significance types

## Troubleshooting

### PNG Export Issues
If automatic PNG export fails:
1. Verify selenium installation: `pip list | grep selenium`
2. Check browser drivers in PATH
3. Use manual export methods instead

### File Not Found Errors
Ensure you're in the correct directory:
```bash
cd /home/matthias/ballpushing_utils/src/PCA
```

### Memory Issues
For large datasets, consider:
- Using the genotype averages version instead of individual flies
- Reducing the number of components shown

## Next Steps

### For Publications
1. Use HTML versions for exploration and analysis
2. Export to PDF via browser for high-quality figures
3. Convert PDF to EPS/SVG for final publication

### For Presentations
1. Use HTML versions directly in presentations (if supported)
2. Export high-resolution screenshots
3. Use the legend PNG file alongside the plots

## File Organization
```
/src/PCA/
├── interactive_scatterplot_matrix_pca.py                      # Main static PCA script (genotype averages)
├── interactive_scatterplot_matrix_temporal_pca.py             # Main temporal PCA script (genotype averages)
├── interactive_scatterplot_matrix_individual.py               # Individual flies static PCA
├── interactive_scatterplot_matrix_temporal_individual.py      # Individual flies temporal PCA
├── create_pca_legends.py                                      # Legend generation
├── interactive_pca_scatterplot_matrix.html                    # Static PCA output (genotype averages)
├── interactive_temporal_pca_scatterplot_matrix.html           # Temporal PCA output (genotype averages)
├── interactive_pca_scatterplot_matrix_individual.html         # Individual flies static PCA output
├── interactive_temporal_pca_scatterplot_matrix_individual.html # Individual flies temporal PCA output
└── pca_legend_combined.png                                    # Combined legend
```

The implementation is complete and ready for use in both research and presentation contexts!
