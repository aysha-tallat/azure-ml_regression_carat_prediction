#!/bin/bash

# Variables
RG_NAME="rg-dp100"
LOCATION="eastus"
WS_NAME="mlws-dp100"
DATA_TRAIN_NAME="diamond_train"
DATA_TEST_NAME="diamond_test"
ENV_NAME="diamonds-env"
COMPUTE_NAME="Aml-Compute"

# Create resource group
az group create --name $RG_NAME --location $LOCATION

# Create workspace
az ml workspace create --name $WS_NAME --resource-group $RG_NAME --location $LOCATION

# Register data asset
az ml data create --name $DATA_TRAIN_NAME --version 1 --path ./diamond_train --type mltable --resource-group $RG_NAME --workspace-name $WS_NAME

# Register data asset
az ml data create --name $DATA_TEST_NAME --version 1 --path ./diamond_test --type mltable --resource-group $RG_NAME --workspace-name $WS_NAME

# Create environments
az ml environment create --name $ENV_NAME --version 1 --conda-file conda.yml --image mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04 --resource-group $RG_NAME --workspace-name $WS_NAME

# Create compute
az ml compute create --name $COMPUTE_NAME --size Standard_DS11_v2 --type amlcompute --min-instances 0 --max-instances 2 --resource-group $RG_NAME --workspace-name $WS_NAME
