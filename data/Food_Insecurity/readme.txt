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

