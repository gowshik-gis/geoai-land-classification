# 🛰️ GeoAI Land Classification System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

> **AI-Powered Satellite Imagery Analysis & Land Cover Mapping Platform**

A complete, end-to-end geospatial AI application that performs automated land use/land cover (LULC) classification using satellite imagery. Built entirely with **free, open-source tools** — no API keys, no Docker, no Ollama required.

---

## 📸 Screenshots

### 🗺️ Interactive Map
![Interactive Map](assets/screenshots/01_interactive_map.png)

### 📊 Analytics Dashboard
![Dashboard](assets/screenshots/02_dashboard.png)

### 🎨 3D Terrain Visualization
![3D Visualization](assets/screenshots/03_3d_visualization.png)

### 📈 Classification Analytics
![Analytics](assets/screenshots/04_analytics.png)

---

## 🏗️ Architecture Overview

![Architecture](assets/diagrams/architecture_overview.png)

### System Components

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Sources** | Sentinel-2, Landsat-9, MODIS, OpenStreetMap | Free satellite imagery |
| **Preprocessing** | NumPy, Rasterio, Xarray | Cloud masking, atmospheric correction |
| **Feature Engineering** | Scikit-learn, Spectral | NDVI, NDWI, texture extraction |
| **ML Model** | Random Forest, Neural Network | Land cover classification |
| **Visualization** | Streamlit, Folium, Plotly | Interactive maps & dashboards |
| **Deployment** | Streamlit Community Cloud | Free hosting |

---

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.10+** — Primary programming language
- **Streamlit** — Web application framework
- **Folium** — Interactive mapping
- **Plotly** — 3D visualizations & charts
- **Scikit-learn** — Machine learning models
- **Pandas / NumPy** — Data processing
- **Matplotlib** — Static visualizations

### Free Data Sources
- **Sentinel-2** (ESA) — 10m resolution multispectral imagery
- **Landsat-9** (NASA/USGS) — 30m resolution thermal & optical
- **MODIS** (NASA) — Daily global coverage
- **OpenStreetMap** (OSM) — Free base maps & vector data

---

## 🚀 Live Demo

**🔗 [View Live Application](https://geoai-land-classification.streamlit.app/)**


---

## 📋 Setup Instructions

### Prerequisites
- Python 3.10 or higher installed
- VS Code (Visual Studio Code) installed
- Git installed
- GitHub account (free)

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/geoai-land-classification.git
cd geoai-land-classification
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`

---

## 🎯 Project Purpose

This project addresses critical environmental monitoring challenges:

- **🌳 Deforestation Monitoring** — Track forest loss in real-time
- **🏙️ Urban Growth Analysis** — Monitor city expansion patterns
- **💧 Water Resource Management** — Detect changes in water bodies
- **🌾 Agricultural Assessment** — Evaluate crop health and land use
- **🌊 Disaster Response** — Rapid damage assessment after floods/fires
- **📊 Environmental Reporting** — Generate automated analytics reports

### Key Features
- ✅ **No API Keys Required** — Uses free, open data sources
- ✅ **Interactive Maps** — Draw, measure, and explore regions
- ✅ **3D Terrain Visualization** — Digital elevation models
- ✅ **Real-time Dashboards** — Live metrics and charts
- ✅ **Change Detection** — Compare multi-temporal imagery
- ✅ **Export Capabilities** — CSV, Excel, PDF reports
- ✅ **Mobile Responsive** — Works on all devices

---

## 📊 Sample Outputs

### Maps
![Land Cover Map](assets/maps/land_cover_classification.png)
*Land Cover Classification Map — 7 classes identified*

![NDVI Map](assets/maps/ndvi_heatmap.png)
*NDVI Vegetation Health Heatmap*

![Change Detection](assets/maps/change_detection.png)
*Multi-temporal Change Detection (2018-2024)*

### Dashboards
![Main Dashboard](assets/dashboards/main_dashboard.png)
*Real-time Analytics Dashboard with KPIs and Charts*

![Analytics Dashboard](assets/dashboards/analytics_dashboard.png)
*Advanced Analytics with Performance Metrics*

### 3D Visualization
![3D Terrain](assets/3d/3d_terrain_dem.png)
*3D Digital Elevation Model with Contours*

![3D Bars](assets/3d/3d_land_cover_bars.png)
*3D Land Cover Classification Bars*

![3D Spectral](assets/3d/3d_spectral_scatter.png)
*3D Spectral Feature Space Clustering*

### Analytics Output
![Accuracy Metrics](assets/analytics/accuracy_metrics.png)
*Classification Accuracy & Performance Metrics*

![Change Analysis](assets/analytics/change_detection_analysis.png)
*Land Cover Change Detection Analysis*

### Workflow Diagram
![Workflow](assets/workflow/workflow_diagram.png)
*Complete Project Workflow from Setup to Deployment*

---

## 📁 Project Structure

```
geoai-land-classification/
├── 📄 app.py                          # Main Streamlit application
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # Project documentation
├── 📄 .gitignore                      # Git ignore rules
│
├── 📁 assets/
│   ├── 📁 screenshots/                # UI screenshots
│   │   ├── 01_interactive_map.png
│   │   ├── 02_dashboard.png
│   │   ├── 03_3d_visualization.png
│   │   └── 04_analytics.png
│   ├── 📁 diagrams/                   # Architecture diagrams
│   │   └── architecture_overview.png
│   ├── 📁 maps/                       # Map visualizations
│   │   ├── land_cover_classification.png
│   │   ├── ndvi_heatmap.png
│   │   └── change_detection.png
│   ├── 📁 dashboards/                 # Dashboard outputs
│   │   ├── main_dashboard.png
│   │   └── analytics_dashboard.png
│   ├── 📁 3d/                         # 3D visualizations
│   │   ├── 3d_terrain_dem.png
│   │   ├── 3d_land_cover_bars.png
│   │   └── 3d_spectral_scatter.png
│   ├── 📁 analytics/                  # Analytics outputs
│   │   ├── accuracy_metrics.png
│   │   └── change_detection_analysis.png
│   └── 📁 workflow/                   # Workflow diagrams
│       └── workflow_diagram.png
│
├── 📁 data/                           # Data directory
│   ├── 📁 raw/                        # Raw satellite imagery
│   ├── 📁 processed/                  # Processed data
│   └── 📁 sample/                     # Sample datasets
│
├── 📁 notebooks/                      # Jupyter notebooks
│   └── exploratory_analysis.ipynb
│
├── 📁 src/                            # Source code
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_extractor.py
│   ├── model.py
│   └── visualizer.py
│
├── 📁 models/                         # Trained models
│   └── random_forest_model.pkl
│
└── 📁 static/                         # Static assets
    └── style.css
```

---

## 🧠 How It Works

### 1. Data Acquisition
- Access free satellite imagery from Sentinel-2, Landsat-9, or MODIS
- Download multispectral bands (RGB, NIR, SWIR)
- Load OpenStreetMap base layers

### 2. Preprocessing
- Apply cloud masking algorithms
- Perform atmospheric correction
- Resample to target resolution (10m - 100m)
- Georeference and project to standard CRS

### 3. Feature Engineering
- Calculate spectral indices:
  - **NDVI** = (NIR - Red) / (NIR + Red) — Vegetation health
  - **NDWI** = (NIR - SWIR) / (NIR + SWIR) — Water content
  - **NDBI** = (SWIR - NIR) / (SWIR + NIR) — Built-up areas
- Extract texture features (GLCM)
- Generate training samples

### 4. Model Training
- Train Random Forest classifier
- Hyperparameter tuning with cross-validation
- Evaluate with confusion matrix and accuracy metrics
- Save trained model for inference

### 5. Classification
- Apply model to entire image
- Generate probability maps
- Post-process with smoothing filters
- Create final land cover map

### 6. Visualization
- Render interactive maps with Folium
- Generate 3D terrain with Plotly
- Build real-time dashboards with Streamlit
- Export reports in multiple formats

---

## 🎓 Learning Outcomes

By working with this project, you will gain expertise in:

- **Remote Sensing** — Satellite imagery processing and analysis
- **Geospatial AI** — Applying ML to geospatial data
- **Data Visualization** — Creating interactive maps and dashboards
- **Python Development** — Building production-ready applications
- **Cloud Deployment** — Hosting apps on free platforms
- **Environmental Monitoring** — Real-world sustainability applications

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **European Space Agency (ESA)** — Sentinel-2 satellite data
- **NASA / USGS** — Landsat and MODIS satellite data
- **OpenStreetMap Contributors** — Free map data
- **Streamlit Team** — Amazing web app framework
- **Scikit-learn Community** — Machine learning tools

---

## 📧 Contact

**Gowshik P** — [@your_linkedin](https://www.linkedin.com/in/gowshik-gis/) — gowshikpiramanayagam@gmail.com

Project Link: [https://github.com/gowshik-gis/geoai-land-classification](https://github.com/gowshik-gis/geoai-land-classification)

---

<p align="center">
  <b>⭐ Star this repo if you found it helpful!</b><br>
  <i>Built with 💚 for a sustainable future</i>
</p>
