#!/usr/bin/env python3

import csv
import numpy as np
import os
import configparser
import pandas as pd
import pickle

path = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()
config.read(os.path.join(path, 'settings.ini'))


def get_items_df():
    items_path=config.get('Datasets', 'items_path')
    return pd.read_csv(items_path)

def get_item_categories_df():
    item_categories_path=config.get('Datasets', 'item_categories_path')
    return pd.read_csv(item_categories_path)
    
def get_shops_df():
    shops_path=config.get('Datasets', 'shops_path')
    return pd.read_csv(shops_path)

def get_sales_df():
    sales_train_path=config.get('Datasets', 'sales_train_path')
    return pd.read_csv(sales_train_path)

def get_test_df():
    sales_test_path=config.get('Datasets', 'test_path')
    return pd.read_csv(sales_test_path)

def save_model(model):
    out_path = config.get('Models', 'model_path')
    pickle.dump(model, open(out_path, "wb"))

def load_model():
    in_path = config.get('Models', 'model_path')
    return pickle.load(open(in_path))