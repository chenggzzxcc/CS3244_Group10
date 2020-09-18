# CS3244project
Repo for collaboration on CS3244 Machine Learning project

## Get started

Make sure you install Git

To clone this repository on your own computer, run

```
git clone https://github.com/Chenggzzxc/CS3244_Group10
```

To be updated with latest changes anyone has contributed run 

```
git pull origin master
```

To publish your new notebook/cleaned data, if you know how to commit and push go ahead, else send it to me (Cheng Ling) and I'll commit and push it.

## Project organisation

This repository is organised as follows:

* Initial data is placed in `data`
* Cleaned data should be placed in `cleaned_data`
* Any processing code (preprocessing, cleaning, etc) that is used to produce data saved into `cleaned_data` should be placed in `processing`. Since it is likely that there will be multiple notebooks for various cleaning/processing, resulting in multiple versions of cleaned data, please annotate each set of [processing code / cleaned data] with the date so we can identify which notebook/script produced the data.
   * Eg. `remove_missing_19_Aug.pynb` is used to produce `data_cleaned_19_Aug.csv`
* Analysis notebooks will be placed in `experimental`, and final results we agree upon should be placed in `model`

With this repo structure, there should be no notebooks in the top level directory and so the path to all data in the notebooks should be `../data/{file name}` or `../cleaned_data/{file name}`