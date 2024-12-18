# # Advanced Python-based Statistical Application

## Overview
This Streamlit-based web application allows users to perform advanced statistical analysis and data visualization on datasets. Users can upload their datasets and interact with tools for data processing, descriptive statistics, inferential statistics, regression analysis, clustering, and advanced features like PCA and time-series analysis.

## Features

### 1. Data Upload
- Upload datasets in **CSV** or **Excel** format.
- Displays the first few rows of the dataset for validation.

### 2. Data Processing
- Handle **missing values** by removing them.
- Rename columns interactively.
- Filter rows based on specific column values.

### 3. Descriptive Statistics
- Generate summary statistics like mean, median, and standard deviation.
- Visualize data distributions using histograms with KDE.

### 4. Inferential Statistics
- Perform **T-tests**, **Chi-Square tests**, and **ANOVA**:
  - **T-test**: Compare two numeric columns.
  - **Chi-Square Test**: Analyze relationships between categorical columns.
  - **ANOVA**: Compare multiple numeric columns.

### 5. Regression Analysis
- **Linear Regression**:
  - Select independent and dependent variables.
  - Visualize regression lines and key metrics (intercept, coefficient).
- **Logistic Regression** (future scope).

### 6. Clustering
- Perform **K-Means Clustering**:
  - Select X and Y-axis variables.
  - Define the number of clusters.
  - Visualize clusters using Plotly scatter plots.

### 7. Advanced Features
- **Principal Component Analysis (PCA)**:
  - Reduce dimensions and visualize the first two principal components.
- **Time-Series Analysis**:
  - Plot time-series data based on a date column and a value column.

## Requirements
- Python 3.8 or later
- Libraries:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `plotly`
  - `scikit-learn`
  - `scipy`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/statistical-app.git
   cd statistical-app
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run statistical_app.py
   ```

4. Open the app in your browser at `http://localhost:8501`.

## Usage
1. **Upload a Dataset**: Use the "Data Upload" page to upload your CSV/Excel file.
2. **Process Data**: Clean and filter data using the "Data Processing" tab.
3. **Analyze**:
   - View summary statistics and distributions in "Descriptive Statistics".
   - Run statistical tests in "Inferential Statistics".
   - Perform regression and clustering analysis in respective sections.
4. **Advanced Tools**: Use PCA and Time-Series analysis for deeper insights.

## Example Dataset
An example dataset is provided for testing. You can also use your custom dataset in the following format:

| Independent_Variable | Dependent_Variable | Cluster_X | Cluster_Y | Date       |
|----------------------|--------------------|-----------|-----------|------------|
| 47.01                | 149.56            | -1.7      | 1.47      | 2023-01-01 |
| 19.32                | 98.40             | -6.4      | -6.0      | 2023-02-01 |

## Known Issues
- **NumPy Compatibility**: Ensure NumPy version `<2.0` to avoid compatibility issues.
   ```bash
   pip install numpy==1.26.0 --force-reinstall
   ```

## Future Enhancements
- Add support for logistic regression.
- Include interactive feature importance metrics.
- Export results and visualizations as reports.

## License
This project is licensed under the MIT License.

## Contact
For issues, suggestions, or contributions:
- **Name**: Thanuka
- **Email**: [your-email@example.com](thanuka.ellepola@gmail,com)

---
Enjoy analyzing your data with ease! 🚀
