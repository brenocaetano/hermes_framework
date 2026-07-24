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


About the Food Insecurity Dataset
The dataset_Food_Insecurity.csv file is an integrated and derived dataset built from the combination of IPVS, RAIS, and CAISAN data. The information was harmonized and aggregated at the district level for the municipality of São Paulo.
Each row represents a district, identified by cod_dist and described by nome_area. The final dataset includes district-level indicators related to population, households, average household income, street markets, food establishments, and establishment densities per 10,000 inhabitants.
To read dataset_Food_Insecurity.csv, use ; as the field separator and , as the decimal separator.
Preserved Information
The dataset retains consolidated district-level indicators, including:
•	Number of households and inhabitants
•	Average household income
•	Percentile group based on the density of healthy food establishments
•	Density of in natura, mixed, and ultra-processed food establishments
•	Number of street markets
•	Total number of food establishments
•	Number of establishments by CNAE category, such as hypermarkets, supermarkets, minimarkets, butcher shops, fruit and vegetable retailers, restaurants, bars, and snack bars
•	Counts of establishments classified as in natura, mixed, and ultra-processed
Information Not Available
The final file does not provide, as individual columns or microdata, part of the original information used to integrate the source datasets.
The following information was removed or was not retained in the final file:
•	Detailed geometries and spatial identifiers of census tracts, such as geometry, AREA, CDGEODI, ID, TIPO, and CDGEODM
•	Original administrative identifiers, such as state code, municipality code, and municipality name associated with the source datasets
•	Auxiliary neighborhood identifiers, such as BairrosSP, BairrosFortaleza, and BairrosRJ
•	Individual RAIS establishment attributes, including postal code, legal nature, establishment type, establishment size, employment links, participation indicators in the Workers' Food Program (PAT), Simples tax-regime indicator, negative RAIS indicator, and complete economic-activity classifications
•	Original lists of establishments and their individual CNAE classifications
•	Geometries and records required to reconstruct the exact association between each census tract, establishment, and district
•	Municipal or rural-context variables that were not retained as independent district-level indicators, including V38, V39, and V44 through V49
Supplementary Documentation
The file dataDescription.pdf provides a description of the variables used during the construction of the dataset.

