# ForecastMitr

A comprehensive Python-based weather forecasting and data analysis platform that combines machine learning capabilities with interactive data visualization and dashboarding.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

ForecastMitr is an advanced weather forecasting system designed to process, analyze, and visualize meteorological data. The project integrates machine learning models with real-time data processing to provide accurate weather predictions and comprehensive data analytics capabilities.

The platform leverages climate data from ECMWF (European Centre for Medium-Range Weather Forecasts) and combines it with machine learning algorithms to generate actionable weather insights.

## ✨ Features

- **Data Processing**: Advanced ETL pipelines for weather data ingestion and preprocessing
- **Machine Learning Models**: Multiple ML algorithms for weather forecasting and prediction
- **Interactive Dashboards**: Real-time visualization using Streamlit
- **Jupyter Notebook Support**: Exploratory data analysis and model experimentation
- **Data Formats**: Support for NetCDF, GRIB, and standard tabular formats
- **Parallel Processing**: Efficient handling of large-scale meteorological datasets
- **Cloud Integration**: AWS S3 support for data storage and retrieval
- **Multi-model Ensemble**: Combine multiple forecasting approaches for improved accuracy

## 📁 Project Structure

```
ForecastMitr/
├── data/                 # Raw and processed weather data
├── models/              # Machine learning models and model artifacts
├── dashboard/           # Streamlit dashboard and web interfaces
├── scripts/             # Utility and processing scripts
├── requirements.lock.txt # Locked dependency versions
├── .gitignore          # Git ignore file
└── README.md           # This file
```

### Directory Descriptions

| Directory | Purpose |
|-----------|---------|
| `data/` | Storage for raw meteorological data, processed datasets, and cache files |
| `models/` | Pre-trained models, model checkpoints, and serialized artifacts |
| `dashboard/` | Streamlit applications for interactive data visualization and real-time monitoring |
| `scripts/` | Python scripts for data preprocessing, model training, and batch processing |

## 📦 Requirements

### System Requirements
- Python 3.8 or higher
- Minimum 4GB RAM (8GB+ recommended for large datasets)
- 2GB free disk space

### Key Dependencies

#### Data Processing & Analysis
- **pandas** (3.0.5) - Data manipulation and analysis
- **numpy** (2.4.6) - Numerical computing
- **scipy** (1.17.1) - Scientific computing
- **xarray** (2026.7.0) - N-dimensional labeled arrays
- **netCDF4** (1.7.4) - NetCDF format support
- **cfgrib** (0.9.15.1) - GRIB data format support
- **eccodes** (2.48.0) - ECMWF data codec

#### Machine Learning
- **scikit-learn** (1.9.1) - Classical ML algorithms
- **xgboost** (3.2.0) - Gradient boosting
- **joblib** (1.6.0) - Efficient serialization

#### Visualization & Dashboard
- **streamlit** (1.63.0) - Interactive web dashboards
- **plotly** (7.0.0) - Interactive plots
- **altair** (6.2.2) - Declarative visualization
- **matplotlib** (3.11.2) - Static plots
- **seaborn** (0.13.2) - Statistical visualization

#### Climate Data Access
- **cdsapi** (0.7.7) - Climate Data Store API
- **ecmwf-datastores-client** (0.5.3) - ECMWF data access

#### Utilities
- **requests** (2.34.2) - HTTP requests
- **click** (8.5.0) - CLI framework
- **tqdm** (4.70.1) - Progress bars
- **PyYAML** (6.0.3) - Configuration files

#### Jupyter & Development
- **jupyter** (1.1.1) - Notebook environment
- **jupyterlab** (4.6.3) - Enhanced notebook interface
- **ipython** (9.17.1) - Interactive shell

For a complete list of dependencies, see [requirements.lock.txt](requirements.lock.txt).

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/S-S-Srichaitanya-angara/ForecastMitr.git
cd ForecastMitr
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.lock.txt
```

Or, if you prefer more flexibility with package versions:

```bash
pip install -e .
```

### 4. Set Up Configuration

Create a `.env` file in the project root for configuration:

```bash
# Climate Data Store credentials
CDS_API_KEY=your_api_key_here

# AWS Configuration (if using S3)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# Data paths
DATA_PATH=./data
MODEL_PATH=./models
```

## 📖 Usage

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`

### Running Data Processing Scripts

```bash
# Process raw meteorological data
python scripts/process_weather_data.py

# Train forecasting models
python scripts/train_models.py

# Generate predictions
python scripts/forecast.py
```

### Using Jupyter Notebooks

```bash
jupyter lab
```

Navigate to the notebooks directory and open analysis files for exploratory data analysis and model experimentation.

### Command Line Interface

```bash
# Get help on available commands
python scripts/cli.py --help

# Process specific weather stations
python scripts/cli.py process --stations station1,station2 --date 2024-01-15

# Run forecasting model
python scripts/cli.py forecast --location "latitude,longitude" --days 7
```

## 🛠 Technologies

### Core Stack
- **Python 3.8+** - Primary programming language
- **Pandas/NumPy** - Data manipulation and numerical computing
- **Scikit-learn/XGBoost** - Machine learning algorithms
- **Streamlit** - Interactive dashboarding

### Data Formats
- **NetCDF** - Hierarchical scientific data format
- **GRIB** - Gridded Binary data format (meteorological standard)
- **CSV/Parquet** - Tabular formats

### Data Sources
- **ECMWF Climate Data Store** - European weather data
- **AWS S3** - Cloud data storage
- **Local file systems** - Direct data access

### Deployment
- **Streamlit Cloud** - Dashboard hosting
- **Docker** - Container-based deployment (optional)
- **AWS Lambda** - Serverless processing (optional)

## ⚙️ Configuration

### Environment Variables

```env
# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/forecast.log

# Data Processing
BATCH_SIZE=100
NUM_WORKERS=4
DATA_CACHE_SIZE=1000

# Model Configuration
MODEL_TYPE=ensemble
ENSEMBLE_MODELS=xgboost,random_forest,linear_regression
PREDICTION_HORIZON=7

# API Configuration
API_TIMEOUT=30
MAX_RETRIES=3
```

### Configuration Files

Place YAML configuration files in the project root:

```yaml
# config.yaml
data:
  source: cds  # or 'local', 's3'
  region: europe
  variables:
    - temperature
    - precipitation
    - wind_speed

models:
  training:
    test_size: 0.2
    validation_size: 0.1
  parameters:
    xgboost:
      max_depth: 6
      learning_rate: 0.1
```

## 📊 Data Pipeline

```
Raw Data → Processing → Feature Engineering → Model Training → Prediction → Dashboard
   ↓          ↓              ↓                    ↓                 ↓
  ECMWF    Cleaning      Feature Store      Validation        Visualization
  S3       Aggregation   Selection          Evaluation
  Local    Interpolation Scaling            Metrics
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.lock.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black .

# Lint code
pylint scripts/ models/
```

## 📝 License

This project is open source. See the LICENSE file for details.

## 📞 Support & Contact

For questions, issues, or suggestions:
- Open an [Issue](https://github.com/S-S-Srichaitanya-angara/ForecastMitr/issues)
- Start a [Discussion](https://github.com/S-S-Srichaitanya-angara/ForecastMitr/discussions)
- Contact the maintainers

## 🙏 Acknowledgments

- **ECMWF** - European Centre for Medium-Range Weather Forecasts for climate data
- **Streamlit** - Interactive web app framework
- **Open-source community** - All the amazing libraries and tools used in this project

---

**Last Updated**: September 2026

**Repository**: [ForecastMitr](https://github.com/S-S-Srichaitanya-angara/ForecastMitr)
