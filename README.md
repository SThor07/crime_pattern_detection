# Crime Hotspot Detection Using Graph Neural Networks

This project predicts urban crime hotspots in Phoenix, AZ using geospatial analysis and Graph Neural Networks (GNNs) in PyTorch Geometric. We process crime data by ZIP code, create a spatial graph based on neighborhood proximity (Haversine distance), and train a Graph Attention Network (GAT) to classify each region as a hotspot or not. Results are visualized on interactive Folium maps.

---

## Quick Start

1. **Clone repo and install dependencies**
    ```bash
    git clone https://github.com/yourusername/crime-hotspot-gnn.git
    cd crime-hotspot-gnn
    pip install torch torch_geometric haversine pandas numpy scikit-learn folium matplotlib
    ```

2. **Download data**
    - [Phoenix Police Department Open Data](https://www.phoenixopendata.com/dataset/crime-incidents)
    - Place CSV in `data/` or update the notebook path.

3. **Run notebook**
    ```bash
    jupyter notebook crime-pattern-detec.ipynb
    ```

---

## Features

- **Data cleaning & geospatial processing**: Cleans and transforms real crime incident data by ZIP and date.
- **Graph creation**: Builds spatial graphs modeling neighborhood proximity.
- **GNN modeling**: GAT predicts crime hotspot status for each region.
- **Evaluation & visualization**: Model evaluated with accuracy, confusion matrix, ROC; results mapped interactively.

---

Academic use only · Arizona State University · CSE 572
