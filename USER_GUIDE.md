# Smart Tender Evaluation - User Guide

## 📖 Quick Start Guide

This guide will walk you through using the Smart Tender Evaluation System step by step.

---

## 🚀 Getting Started

### Step 1: Launch the Application
1. Open your terminal/command prompt
2. Navigate to the project directory
3. Run: `streamlit run app.py`
4. The app will open in your browser at `http://localhost:8501`

### Step 2: Prepare Your Data
1. **Download Sample Data**: Click "Download Sample CSV Schema" in the sidebar
2. **Format Your Data**: Use the sample as a template
3. **Required Columns**: Ensure you have at least `supplier`, `price`, `quality`, `delivery_time_days`
4. **Save as CSV**: Save your data in CSV format

### Step 3: Upload Your Data
1. Click "Upload Tender Offers CSV" in the sidebar
2. Select your CSV file
3. The app will automatically process and validate your data

---

## 📊 Dashboard Tab

### Overview
The Dashboard provides a comprehensive overview of your tender data with key metrics and visualizations.

### Features Available

#### 1. **KPI Metrics**
- **Min Price**: Lowest offer price
- **Max Price**: Highest offer price  
- **Mean Price**: Average offer price
- **Median Price**: Middle offer price

#### 2. **Supplier Comparison Charts**
- **Price Comparison**: Bar chart showing suppliers ranked by price (best deals first)
- **Quality Comparison**: Bar chart showing suppliers ranked by quality score
- **Color Coding**: Charts use color gradients to highlight best/worst performers

#### 3. **Radar Chart (Supplier Profiles)**
- **Multi-dimensional View**: Shows suppliers across multiple variables
- **Normalized Data**: All variables scaled to 0-1 for fair comparison
- **Interactive**: Hover over points to see exact values

#### 4. **Total Cost Analysis**
- **Calculation**: Price - Discount + Shipping Cost
- **Table View**: Shows breakdown for each supplier
- **Best Value**: Identifies suppliers with lowest total cost

### How to Use Dashboard
1. **Upload your data** first
2. **Review KPI metrics** to understand price distribution
3. **Analyze charts** to identify best suppliers
4. **Use radar chart** to compare supplier profiles
5. **Check total cost** for complete cost analysis

---

## 🎯 Scoring Model Tab

### Overview
The Scoring Model allows you to rank suppliers using different weighting scenarios and custom criteria.

### Available Scenarios

#### 1. **Equal Weights**
- **Description**: All variables weighted equally
- **Use Case**: When all criteria are equally important
- **Formula**: Each variable gets 1/n weight (where n = number of variables)

#### 2. **Price-Focused**
- **Description**: 60% weight on price, 40% distributed among other variables
- **Use Case**: When cost is the primary concern
- **Formula**: Price = 0.6, Others = 0.4/(n-1)

#### 3. **Quality-Focused**
- **Description**: 60% weight on quality, 40% distributed among other variables
- **Use Case**: When quality is the primary concern
- **Formula**: Quality = 0.6, Others = 0.4/(n-1)

#### 4. **Custom Weights**
- **Description**: Set your own weights for each variable
- **Use Case**: When you have specific requirements
- **Features**: 
  - Interactive sliders for each variable
  - Automatic normalization if weights don't sum to 1.0
  - Real-time weight validation

### How Scoring Works

#### Normalization Process
1. **Price Variables**: Normalized for minimization (lower is better)
2. **Other Variables**: Normalized for maximization (higher is better)
3. **Formula**: (value - min) / (max - min) or (max - value) / (max - min)

#### Final Score Calculation
```
Score = Σ(weight_i × normalized_value_i)
```

### How to Use Scoring Model
1. **Select a scenario** from the dropdown
2. **For custom weights**:
   - Adjust sliders for each variable
   - Ensure weights sum to 1.0 (or close to it)
   - Review the normalized weights display
3. **Review results**:
   - See ranked suppliers table
   - Note the best supplier and their score
   - Analyze score distribution

---

## 🔍 SSBI (Self-Service Business Intelligence) Tab

### Overview
SSBI provides advanced business intelligence tools for deeper analysis of your tender data.

### Available Features

#### 1. **Price Summary KPIs**
- **Min/Max/Mean/Median/Std Dev**: Statistical summary of prices
- **Purpose**: Understand price distribution and variability

#### 2. **Score Composition**
- **Stacked Bar Chart**: Shows how each variable contributes to final scores
- **Purpose**: Understand which variables drive supplier rankings
- **Interactive**: Hover to see exact contribution values

#### 3. **Price Distribution**
- **Box Plot**: Shows price distribution with outliers
- **Supplier Overlay**: Individual supplier points on the distribution
- **Purpose**: Identify price outliers and understand distribution

#### 4. **Supplier Profiles (Radar Chart)**
- **Multi-dimensional Comparison**: Compare suppliers across all variables
- **Normalized View**: All variables scaled to 0-1
- **Purpose**: Visual comparison of supplier strengths/weaknesses

#### 5. **Price vs Quality/Technical**
- **Scatter Plot**: Shows relationship between price and quality
- **Color Coding**: Suppliers colored by name
- **Purpose**: Identify value propositions (high quality, low price)

#### 6. **Supplier Comparison Table**
- **Interactive Table**: All supplier data in tabular format
- **Sortable**: Click column headers to sort
- **Purpose**: Detailed comparison of all variables

#### 7. **Correlation Heatmap**
- **Variable Relationships**: Shows correlations between all numeric variables
- **Color Scale**: Red (negative) to Blue (positive) correlations
- **Purpose**: Understand variable relationships and dependencies

#### 8. **Pareto Chart**
- **Price Contribution**: Shows cumulative price contribution by supplier
- **80/20 Analysis**: Identify suppliers contributing most to total cost
- **Purpose**: Focus on high-impact suppliers

#### 9. **Filter Suppliers**
- **Interactive Filters**: Sliders for each variable
- **Real-time Filtering**: See filtered results immediately
- **Purpose**: Focus analysis on specific supplier segments

### How to Use SSBI
1. **Start with KPIs** to understand your data
2. **Use score composition** to understand ranking drivers
3. **Analyze distributions** to identify patterns
4. **Compare suppliers** using radar charts
5. **Filter data** to focus on specific segments
6. **Export results** for further analysis

---

## 🧠 Advanced Analytics Tab

### Overview
Advanced Analytics uses machine learning to provide deeper insights and identify patterns in your data.

### Available Features

#### 1. **Anomaly Detection**
- **Algorithm**: Isolation Forest
- **Purpose**: Identify suspicious or outlier offers
- **Output**: 
  - Table of detected anomalies
  - Scatter plot showing outliers
  - Risk assessment for each supplier

#### 2. **Principal Component Analysis (PCA)**
- **Algorithm**: PCA with 2 components
- **Purpose**: Reduce dimensionality for visualization
- **Output**: 2D scatter plot of suppliers
- **Use Case**: Identify supplier clusters and patterns

#### 3. **Feature Importance**
- **Algorithm**: Random Forest Regressor
- **Purpose**: Determine which variables most affect price
- **Output**: 
  - Bar chart of feature importance
  - Ranked list of variables
- **Use Case**: Understand price drivers

#### 4. **Procurement Risk Scoring**
- **Risk Types**: 7 different procurement risks
- **Algorithms**: Statistical and ML-based risk assessment
- **Output**: 
  - Risk scores for each supplier
  - Risk breakdown table
  - Risk visualization

### Risk Types Explained

#### 1. **Lowball Risk**
- **Definition**: Suspiciously low prices
- **Calculation**: (mean_price - price) / mean_price where price < threshold
- **Threshold**: 85% of mean price

#### 2. **Drip Pricing Risk**
- **Definition**: Hidden cost indicators
- **Calculation**: Price normalization × (1 - quality normalization)
- **Purpose**: Identify suppliers with hidden costs

#### 3. **Drip Pricing Flag**
- **Definition**: High price, low quality combinations
- **Calculation**: Binary flag for price > 75th percentile AND quality < 25th percentile
- **Purpose**: Identify poor value propositions

#### 4. **Market Signaling Risk**
- **Definition**: High price, high score combinations
- **Calculation**: Binary flag for price rank > 80% AND score rank > 80%
- **Purpose**: Identify potential market signaling

#### 5. **Cover Bid Risk**
- **Definition**: High price, low score combinations
- **Calculation**: Binary flag for price > 75th percentile AND score < median
- **Purpose**: Identify non-serious bids

#### 6. **Decoy Bid Risk**
- **Definition**: Statistical outliers
- **Calculation**: Z-score > 2 for price or score
- **Purpose**: Identify unusual bids

#### 7. **Bid Similarity Risk**
- **Definition**: Potential collusion indicators
- **Calculation**: Similar rounded prices or scores
- **Purpose**: Identify potential bid rigging

### How to Use Advanced Analytics
1. **Run scoring model first** to generate scores
2. **Review anomaly detection** to identify suspicious offers
3. **Use PCA** to understand supplier clustering
4. **Analyze feature importance** to understand price drivers
5. **Review risk scores** to assess supplier reliability
6. **Take action** based on insights

---

## 💼 Negotiation Strategy Tab

### Overview
Negotiation Strategy provides tools to optimize your negotiation approach and identify the best suppliers for negotiation.

### Available Features

#### 1. **Variable Impact Analysis**
- **Algorithm**: Random Forest feature importance
- **Purpose**: Understand which variables most affect selection scores
- **Output**: Bar chart of variable importance
- **Use Case**: Focus negotiation on most impactful variables

#### 2. **Trade-Off Matrix**
- **Visualization**: 3D scatter plot
- **Variables**: Price, Quality, Delivery
- **Color Coding**: Winners vs non-winners
- **Purpose**: Visualize trade-offs between key variables

#### 3. **Ranked Supplier Table**
- **Scoring**: Negotiation score = quality / (price × (1 + 0.01 × delivery))
- **Ranking**: Suppliers ranked by negotiation score
- **Winner Marking**: Top suppliers marked as winners
- **Purpose**: Identify best suppliers for negotiation

#### 4. **Intelligent Negotiation Advice**
- **Algorithm**: Benchmarking against optimal set
- **Output**: Specific recommendations for each supplier
- **Advice Types**:
  - Price reduction recommendations
  - Quality improvement suggestions
  - Delivery time optimization
- **Purpose**: Provide actionable negotiation guidance

### How to Use Negotiation Strategy
1. **Review variable impact** to understand negotiation priorities
2. **Analyze trade-off matrix** to understand relationships
3. **Identify winners** from ranked table
4. **Review negotiation advice** for specific recommendations
5. **Plan negotiation strategy** based on insights

---

## 🔮 What-If Analysis Tab

### Overview
What-If Analysis allows you to simulate changes to supplier parameters and see how they affect rankings.

### Available Features

#### 1. **Impactful Variables Identification**
- **Algorithm**: Random Forest feature importance
- **Output**: Top 5 most impactful variables
- **Purpose**: Focus analysis on variables that matter most

#### 2. **Scenario Simulation**
- **Interactive Sliders**: Adjust values for each variable
- **Real-time Updates**: See ranking changes immediately
- **Supplier Selection**: Choose which supplier to simulate
- **Purpose**: Test different scenarios

#### 3. **Impact Assessment**
- **Score Changes**: See how scores change with parameter adjustments
- **Ranking Changes**: See how rankings shift
- **Sensitivity Analysis**: Identify most sensitive variables
- **Purpose**: Understand parameter sensitivity

### How to Use What-If Analysis
1. **Select a supplier** to simulate changes for
2. **Review current values** and rankings
3. **Adjust sliders** to simulate changes
4. **Observe impact** on scores and rankings
5. **Identify sensitivities** to understand key variables
6. **Plan strategies** based on insights

---

## 📄 PDF Report Generation

### Overview
Generate professional PDF reports with all your analyses and visualizations.

### Available Report Sections

#### 1. **Dashboard**
- KPI metrics
- Supplier comparison charts
- Radar charts
- Total cost analysis

#### 2. **Scoring Model**
- Scenario results
- Supplier rankings
- Score distributions

#### 3. **SSBI**
- Price summaries
- Score compositions
- Correlation analysis
- Filtered results

#### 4. **Advanced Analytics**
- Anomaly detection results
- PCA analysis
- Feature importance
- Risk scoring

#### 5. **Negotiation Strategy**
- Variable impact analysis
- Trade-off matrix
- Negotiation rankings
- Strategic advice

#### 6. **What-If Analysis**
- Impactful variables
- Scenario simulations
- Sensitivity analysis

### How to Generate Reports
1. **Select sections** you want to include
2. **Click "Generate PDF Report"**
3. **Wait for processing** (progress bar will show status)
4. **Download the report** when complete
5. **Share with stakeholders**

### Report Features
- **Professional Format**: Cover page, table of contents
- **Executive Summary**: High-level insights
- **Charts & Tables**: All visualizations included
- **Risk Explanations**: Detailed risk descriptions
- **Callouts & Tips**: Highlighted important information

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. **App Won't Start**
- **Problem**: Import errors or missing dependencies
- **Solution**: Install required packages
  ```bash
  pip install streamlit pandas numpy plotly altair fpdf2 qrcode pillow scikit-learn
  ```

#### 2. **Data Won't Load**
- **Problem**: CSV format issues
- **Solution**: 
  - Check CSV format (comma-separated)
  - Ensure required columns are present
  - Verify data types (numeric for numbers)

#### 3. **Charts Not Displaying**
- **Problem**: Browser or JavaScript issues
- **Solution**:
  - Use Chrome or Firefox
  - Enable JavaScript
  - Clear browser cache

#### 4. **Slow Performance**
- **Problem**: Large datasets or memory issues
- **Solution**:
  - Reduce data size
  - Use reset button to clear cache
  - Restart the application

#### 5. **PDF Generation Fails**
- **Problem**: Missing dependencies or memory issues
- **Solution**:
  - Install fpdf2: `pip install fpdf2`
  - Reduce data size
  - Check available memory

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing package | Install with pip |
| `ValueError: No numeric data` | Invalid data types | Check CSV format |
| `KeyError: 'supplier'` | Missing column | Add supplier column |
| `MemoryError` | Large dataset | Reduce data size |

---

## 💡 Tips and Best Practices

### Data Preparation
1. **Use consistent formats** for all data
2. **Include all required columns** (supplier, price, quality, delivery_time_days)
3. **Use appropriate data types** (numbers for numeric fields)
4. **Check for missing values** and handle appropriately
5. **Validate data** before uploading

### Analysis Workflow
1. **Start with Dashboard** to understand your data
2. **Use Scoring Model** to rank suppliers
3. **Explore SSBI** for detailed analysis
4. **Apply Advanced Analytics** for deeper insights
5. **Develop Negotiation Strategy** based on findings
6. **Run What-If Analysis** to test scenarios
7. **Generate PDF Report** for stakeholders

### Interpretation Guidelines
1. **Consider context** when interpreting results
2. **Look for patterns** across multiple analyses
3. **Validate insights** with domain knowledge
4. **Consider risk factors** in decision making
5. **Use multiple scenarios** to test robustness

### Performance Optimization
1. **Limit data size** for faster processing
2. **Use caching** effectively
3. **Clear cache** when needed
4. **Restart app** if performance degrades

---

## 📞 Getting Help

### Support Resources
1. **Documentation**: Check this user guide and README
2. **Sample Data**: Use the provided sample CSV
3. **Error Messages**: Check the troubleshooting section
4. **Developer Contact**: Reach out to Aryan Zabihi

### Contact Information
- **LinkedIn**: [Aryan Zabihi](https://www.linkedin.com/in/aryanzabihi/)
- **GitHub**: [Aryanzabihi](https://github.com/Aryanzabihi)
- **Email**: Available through LinkedIn

### Feedback and Improvements
- **Report Bugs**: Use GitHub issues
- **Feature Requests**: Submit through GitHub
- **Documentation**: Suggest improvements to this guide

---

*This user guide is maintained by Aryan Zabihi. For the latest updates, please check the GitHub repository.* 