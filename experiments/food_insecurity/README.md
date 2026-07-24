# HERMES - Food Insecurity Analysis and Confidence Labels

This repository contains the notebooks responsible for conducting machine learning experiments and generating visualizations on data related to urban food insecurity. The HERMES framework was structured to evaluate multimodal samples and generate confidence metrics (margins, entropy, flip-rate, and agreement rates).

## Repository Structure

The repository consists of two main notebooks:

### 1. `hermes.ipynb`

* **Function:** This is the computational engine of the experiments.
* **What it does:** It is responsible for loading the databases, labeling the quantiles (using algorithms like Label Spreading), calculating distance matrices (Levenshtein), and generating perturbed matrices to validate labels and obtain the model's topological and logical metrics.
* **Outputs:** All distance matrices, lists of instances divided into quantiles, calculated rates, NDS ranking results, and the raw metadata of the logical perturbations.

### 2. `graphics.ipynb`

* **Function:** A tool dedicated exclusively to the extraction and visual export of the data consolidated by HERMES.
* **What it does:** It reads all the data and tables produced in the experimental stage and generates three-dimensional (and scatter) visualizations focused on the logical metrics of algorithmic resilience.

#### Generated Graphics and Their Directories

To ensure organization, `graphics.ipynb` automatically saves the static and interactive files (such as HTML, PDF, and PNG) into isolated folders created in the execution root, namely:

* **Objective graphics based on NDS:**
* 📁 Saved in: `graficos_plotly_quantis_nsga2`


* **Surface Graphics (Raw Margin Frequency Metric):**
* 📁 Saved in: `graficos_superficie`


* **Surface Graphics (Frequency Metric in Logarithmic Scale):**
* 📁 Saved in: `graficos_superficie_log`


* **Surface Graphics (Coloring based on Flip-Rate via Plotly):**
* 📁 Saved in: `graficos_superficie_log_color_flip`


* **Surface Graphics (Coloring based on Flip-Rate via Matplotlib):**
* 📁 Saved in: `graficos_matplotlib`


* **Surface Graphics (Coloring by Agreement Rate - Global Dataset Scale):**
* 📁 Saved in: `graficos_superficie_soma_concordancia_percentual`


* **Surface Graphics (Coloring by Agreement Rate - Dynamic Scale by Quantile):**
* 📁 Saved in: `graficos_superficie_soma_concordancia_percentual_dinamico`


* **3D Scatter Graphics (Logarithmic Scale):**
* 📁 Saved in: `graficos_dispersao_log`
