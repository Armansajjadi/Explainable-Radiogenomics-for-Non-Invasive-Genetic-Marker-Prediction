# Explainable Radiogenomics for IDH Mutation Prediction

This project aims to predict IDH mutation status in glioma patients using radiogenomics, combining medical imaging data with genomic information. This README covers the initial data preparation and exploratory data analysis (EDA) phases of the project.

## Data Preparation

The initial dataset is loaded from `UCSF-PDGM-metadata_v5.csv`. The following data preparation steps are performed:

  * **Feature Selection:** A new DataFrame is created containing only the 'ID' and 'IDH' columns, as these are the primary features of interest for this stage of the project.
  * **Binarization of Target Variable:** The 'IDH' column, which contains categorical data about the mutation status, is converted into a binary format. A new column, `IDH_binary`, is created where `1` represents a mutation and `0` represents the wildtype.
  * **Final DataFrame:** The original 'IDH' column is dropped, resulting in a final DataFrame ready for analysis with 'ID' and 'IDH\_binary' columns.

## Data Visualization

To understand the distribution of the target variable, a pie chart is generated to visualize the proportion of patients with and without the IDH mutation.

### Brain Scan Visualization

To visually inspect the imaging data, 3D visualizations of the brain scans can be generated. Here is an example of what a 3D brain scan visualization looks like:

![3D Brain Scan](brain_tumor_3rd_person_view.gif)
