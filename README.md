# Smart Tender Evaluation System

https://tender-management.streamlit.app/

## 📊 Overview

The **Smart Tender Evaluation System** is a comprehensive, AI-powered web application designed for procurement professionals to analyze, score, and evaluate supplier offers in tender processes. Built with Streamlit, it provides advanced analytics, machine learning insights, and professional reporting capabilities.

**Author:** Aryan Zabihi  
**Version:** 2.0  
**Last Updated:** December 2024

---

## 🚀 Key Features

### 📈 **Core Functionality**
- **Multi-Scenario Scoring**: Equal weights, price-focused, quality-focused, and custom weight scenarios
- **Advanced Analytics**: Machine learning-powered anomaly detection and risk assessment
- **Interactive Visualizations**: Dynamic charts and graphs using Plotly
- **Professional PDF Reports**: Automated report generation with charts and insights
- **Real-time Data Processing**: Instant analysis of uploaded tender data

### 🎯 **Scoring & Analysis**
- **Supplier Ranking**: Multi-criteria decision analysis
- **Risk Assessment**: Procurement risk scoring (lowballing, drip pricing, etc.)
- **Anomaly Detection**: Isolation Forest algorithm for outlier detection
- **Feature Importance**: Random Forest analysis for variable impact
- **Correlation Analysis**: Heatmaps and relationship analysis

### 📊 **Visualization & Reporting**
- **Dashboard**: KPI metrics and supplier comparisons
- **SSBI**: Self-Service Business Intelligence with interactive charts
- **Advanced Analytics**: PCA, feature importance, and risk scoring
- **Negotiation Strategy**: Trade-off analysis and strategic guidance
- **What-If Analysis**: Scenario modeling and impact assessment

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download
```bash
# If using git
git clone <repository-url>
cd Tender3

# Or download and extract the files
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

---

## 📋 Data Requirements

### Required CSV Format
Your tender data should be in CSV format with the following structure:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `supplier` | Text | Supplier name/identifier | ✅ Yes |
| `price` | Numeric | Offer price | ✅ Yes |
| `quality` | Numeric | Quality score (0-100) | ✅ Yes |
| `delivery_time_days` | Numeric | Delivery time in days | ✅ Yes |
| `warranty_months` | Numeric | Warranty period in months | ✅ Yes |
| `experience_years` | Numeric | Supplier experience in years | ✅ Yes |
| `compliance` | Text | Compliance status (Yes/No) | ✅ Yes |
| `technical` | Numeric | Technical score (0-100) | Optional |
| `certifications` | Text | Certifications held | Optional |
| `country` | Text | Supplier country | Optional |
| `supplier_type` | Text | Manufacturer/Distributor | Optional |

### Sample Data Structure
```csv
supplier,price,quality,delivery_time_days,warranty_months,experience_years,compliance,technical,certifications,country,supplier_type
Supplier A,1000,80,35,24,10,Yes,90,ISO9001;CE,USA,Manufacturer
Supplier B,1175,86,48,18,6,No,87,ISO14001,Germany,Distributor
Supplier C,1080,79,33,20,8,Yes,86,ISO9001,USA,Manufacturer
```

---

## 🎮 How to Use

### 1. **Data Upload**
- Click "Upload Tender Offers CSV" in the sidebar
- Select your CSV file with supplier data
- The app will automatically detect and validate your data

### 2. **Dashboard Analysis**
- **Supplier Comparison**: View price and quality comparisons
- **KPI Metrics**: Min, max, mean, and median prices
- **Radar Charts**: Multi-dimensional supplier profiles
- **Total Cost Analysis**: Price - discount + shipping cost

### 3. **Scoring Model**
- **Select Scenario**: Choose from predefined scenarios or create custom weights
- **Equal Weights**: All variables weighted equally
- **Price-Focused**: 60% weight on price, 40% distributed among other variables
- **Quality-Focused**: 60% weight on quality, 40% distributed among other variables
- **Custom Weights**: Set your own weights for each variable

### 4. **SSBI (Self-Service Business Intelligence)**
- **Score Composition**: Stacked bar charts showing variable contributions
- **Price Distribution**: Box plots and supplier overlays
- **Supplier Profiles**: Radar charts for multi-dimensional comparison
- **Correlation Heatmap**: Variable relationship analysis
- **Pareto Chart**: Price contribution analysis
- **Filter Suppliers**: Interactive filtering by variable ranges

### 5. **Advanced Analytics**
- **Anomaly Detection**: Identify suspicious offers using Isolation Forest
- **PCA Analysis**: Dimensionality reduction for better visualization
- **Feature Importance**: Random Forest analysis of variable impact
- **Risk Scoring**: Comprehensive procurement risk assessment

### 6. **Negotiation Strategy**
- **Variable Impact Analysis**: Feature importance for negotiation
- **Trade-Off Matrix**: 3D visualization of price, quality, and delivery
- **Ranked Supplier Table**: Suppliers ranked by negotiation score
- **Intelligent Advice**: AI-powered negotiation recommendations

### 7. **What-If Analysis**
- **Scenario Modeling**: Simulate changes to supplier parameters
- **Impact Assessment**: See how changes affect rankings
- **Sensitivity Analysis**: Identify most impactful variables

### 8. **PDF Report Generation**
- **Select Sections**: Choose which analyses to include
- **Professional Format**: Cover page, table of contents, and executive summary
- **Charts & Tables**: All visualizations included in PDF
- **Download**: Get a comprehensive report for stakeholders

---

## 🔧 Technical Architecture

### Core Components

#### 1. **Data Processing Layer**
```python
# Utility functions for data handling
def get_numeric_columns(df)
def get_variable_list(df)
def normalize_column(col, minimize=False)
def calculate_risk_scores(df)
```

#### 2. **Scoring Engine**
```python
# Multi-scenario scoring system
def get_weights(variables, scenario)
def apply_common_layout(fig)
```

#### 3. **Visualization Engine**
```python
# Chart generation functions
def score_composition_tab()
def price_distribution_tab()
def supplier_profiles_tab()
def correlation_heatmap_tab()
```

#### 4. **Analytics Engine**
```python
# Machine learning components
def advanced_analytics_tab()
def calculate_risk_scores()
```

#### 5. **PDF Generation System**
```python
# Professional reporting
class PDFReport(FPDF)
def generate_professional_pdf_report()
```

### Machine Learning Models

#### **Anomaly Detection**
- **Algorithm**: Isolation Forest
- **Purpose**: Identify suspicious or outlier offers
- **Parameters**: contamination=0.1, random_state=42

#### **Feature Importance**
- **Algorithm**: Random Forest Regressor
- **Purpose**: Determine variable impact on outcomes
- **Output**: Feature importance scores

#### **Dimensionality Reduction**
- **Algorithm**: Principal Component Analysis (PCA)
- **Purpose**: Visualize high-dimensional data
- **Components**: 2 principal components

### Risk Assessment Framework

The system calculates 7 types of procurement risks:

1. **Lowball Risk**: Suspiciously low prices
2. **Drip Pricing Risk**: Hidden cost indicators
3. **Drip Pricing Flag**: High price, low quality combinations
4. **Market Signaling Risk**: High price, high score combinations
5. **Cover Bid Risk**: High price, low score combinations
6. **Decoy Bid Risk**: Statistical outliers
7. **Bid Similarity Risk**: Potential collusion indicators

---

## 📊 Performance Optimizations

### Caching Strategy
- **@st.cache_data**: Applied to all data processing functions
- **Chart Caching**: Prevents redundant chart generation
- **PDF Caching**: Optimizes report generation

### Memory Management
- **Lazy Loading**: Optional dependencies loaded only when needed
- **Efficient Data Structures**: Optimized pandas operations
- **Garbage Collection**: Proper cleanup of temporary objects

### Processing Speed
- **Vectorized Operations**: Using pandas/numpy vectorized functions
- **Parallel Processing**: Where applicable for large datasets
- **Optimized Algorithms**: Efficient implementation of ML models

---

## 🔍 Troubleshooting

### Common Issues

#### **1. Import Errors**
```bash
# Solution: Install missing dependencies
pip install streamlit pandas numpy plotly altair fpdf2 qrcode pillow scikit-learn
```

#### **2. Data Loading Issues**
- **Check CSV format**: Ensure proper comma separation
- **Column names**: Verify required columns are present
- **Data types**: Ensure numeric columns contain valid numbers

#### **3. Memory Issues**
- **Reduce data size**: Limit to essential columns
- **Clear cache**: Use the reset button in sidebar
- **Restart app**: Close and reopen the application

#### **4. Chart Rendering Issues**
- **Browser compatibility**: Use Chrome or Firefox
- **JavaScript enabled**: Ensure JavaScript is enabled
- **Clear browser cache**: Clear browser cache and cookies

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing dependency | Install required package |
| `ValueError: No numeric data` | Invalid data types | Check CSV format |
| `KeyError: 'supplier'` | Missing required column | Add supplier column |
| `MemoryError` | Large dataset | Reduce data size or restart |

---

## 📈 Advanced Features

### Custom Weight Scenarios
Create your own scoring scenarios by setting custom weights:

```python
# Example custom weights
custom_weights = {
    'price': 0.4,
    'quality': 0.3,
    'delivery_time_days': 0.2,
    'warranty_months': 0.1
}
```

### Risk Threshold Customization
Adjust risk thresholds for your specific needs:

```python
# Modify risk calculation parameters
threshold = mean_price * 0.85  # Lowball threshold
contamination = 0.1  # Anomaly detection sensitivity
```

### PDF Report Customization
Customize PDF reports by selecting specific sections:

```python
# Available report sections
report_options = {
    "Dashboard": True,
    "Scoring Model": True,
    "SSBI": True,
    "Advanced Analytics": True,
    "Negotiation Strategy": True,
    "What-If Analysis": True
}
```

---

## 📁 Project Structure

```
Tender3/
├── app.py                    # Main application file
├── requirements.txt          # Python dependencies
├── README.md                # This documentation
├── USER_GUIDE.md            # Detailed user guide
├── .venv/                   # Virtual environment (if created)
├── .streamlit/              # Streamlit configuration
└── __pycache__/             # Python cache files
```

### File Descriptions

- **`app.py`**: The main application with all features and functionality
- **`requirements.txt`**: List of all required Python packages
- **`README.md`**: This comprehensive documentation
- **`USER_GUIDE.md`**: Step-by-step user instructions

---

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to functions
- Include docstrings for all functions
- Write unit tests for new features

---

## 📞 Support & Contact

### Developer Information
- **Name**: Aryan Zabihi
- **LinkedIn**: [Aryan Zabihi](https://www.linkedin.com/in/aryanzabihi/)
- **GitHub**: [Aryanzabihi](https://github.com/Aryanzabihi)

### Support Channels
- **Issues**: Report bugs via GitHub issues
- **Documentation**: Check this README and USER_GUIDE.md
- **Community**: Join the discussion in GitHub discussions

### Donations
If you find this tool valuable, consider supporting the development:
- **Bitcoin**: `bc1q9mqj6xm22g3g4lvxt97z909gjty223ltn7tl40`

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🔄 Version History

### Version 2.0 (Current)
- ✅ Advanced analytics with machine learning
- ✅ Professional PDF reporting
- ✅ Risk assessment framework
- ✅ Interactive visualizations
- ✅ Performance optimizations
- ✅ Clean project structure

### Version 1.0
- ✅ Basic scoring functionality
- ✅ Simple visualizations
- ✅ CSV data processing

---

## 🎯 Roadmap

### Planned Features
- [ ] Real-time collaboration
- [ ] Database integration
- [ ] API endpoints
- [ ] Mobile app
- [ ] Advanced ML models
- [ ] Integration with ERP systems

### Performance Improvements
- [ ] GPU acceleration
- [ ] Distributed processing
- [ ] Advanced caching
- [ ] Real-time updates

---

## 🚀 Quick Start

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run the app**: `streamlit run app.py`
3. **Upload data**: Use the sample CSV or your own data
4. **Explore features**: Start with Dashboard, then try other tabs
5. **Generate reports**: Use the PDF report feature for stakeholders

---


*This documentation is maintained by Aryan Zabihi. For the latest updates, please check the GitHub repository.* 
