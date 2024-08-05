import os
import re

import pandas
import pyodbc
from scipy.io import loadmat

# Define file_path as a global variable
file_path = None


def get_absolute_path(relative_path):
    """
    Get the absolute file path based on a relative path.

    :param relative_path: Relative path to the file from the project root directory.
    :return: Absolute file path.
    """
    # Get the directory of the current script
    this_file_path = os.path.dirname(__file__)
    # Navigate two levels up
    base_path = os.path.dirname(os.path.dirname(this_file_path))
    # Construct the absolute file path
    return os.path.join(base_path, relative_path)


def etl_mat_file(relative_path, data_item):
    """
    Load data from a Matlab file (.mat)

    Extract:
        Load .mat data

    Transform:
        1. Do not keep metadata, but only specific value data_item

    Load:
        Return DataFrame containing the data

    :param relative_path: Relative path to the .mat file.
    :param data_item: Relative path to the .mat file.
    :return: DataFrame containing the data.
    """
    global file_path
    try:
        # Get the absolute path to the file
        file_path = get_absolute_path(relative_path)

        # Load .mat file
        data = loadmat(str(file_path))

        # return DataFrame containing data using specific value from matlab data[] array
        return pandas.DataFrame(data[data_item])
    except Exception as e:
        raise Exception(f"Error reading .mat file: {e}")

def etl_mat_file_fill_na(relative_path, data_item):
    """
    Load data from a Matlab file (.mat)

    Extract:
        Load .mat data

    Transform:
        1. Do not keep metadata, but only specific value data_item

    Load:
        Return DataFrame containing the data

    :param relative_path: Relative path to the .mat file.
    :param data_item: Relative path to the .mat file.
    :return: DataFrame containing the data.
    """
    global file_path
    try:
        # Get the absolute path to the file
        file_path = get_absolute_path(relative_path)

        # Load .mat file
        data = loadmat(str(file_path))

        # return DataFrame containing data using specific value from matlab data[] array
        return pandas.DataFrame(data[data_item]).fillna(0)
    except Exception as e:
        raise Exception(f"Error reading .mat file: {e}")


def etl_text_file(relative_path, delimiter):
    """
    Load data from a text file (.txt)

    Extract: Load .txt data

    Transform:
        1. Data passed do not have header, so do not use header when reading the file
        2. Use delimiter passed as param to separate values

    Load: Return DataFrame containing the data

    :param relative_path: Relative path to the .txt file.
    :param delimiter: Delimiter used in the .txt file.
    :return: DataFrame containing the data.
    """
    global file_path
    try:
        # Get the absolute path to the file
        file_path = get_absolute_path(relative_path)

        # Start reading the file
        # Use header none, so it does not waste first line of data as header
        # Use delimiter passed as parameter
        # return DataFrame containing data
        return pandas.read_csv(str(file_path), header=None, delimiter=delimiter).fillna(0)
    except Exception as e:
        raise Exception(f"Error reading .txt file: {e}")


def etl_text_file_(relative_path, delimiter, name):
    """
    Load data from a text file (.txt)

    Extract: Load .txt data

    Transform:
        1. Data passed do not have header, so do not use header when reading the file
        2. Use delimiter passed as param to separate values

    Load: Return DataFrame containing the data

    :param relative_path: Relative path to the .txt file.
    :param delimiter: Delimiter used in the .txt file.
    :param name: Delimiter used in the .txt file.
    :return: DataFrame containing the data.
    """
    global file_path
    try:
        # Get the absolute path to the file
        file_path = get_absolute_path(relative_path)

        # Start reading the file
        # Use header none, so it does not waste first line of data as header
        # Use delimiter passed as parameter
        # return DataFrame containing data
        return pandas.read_csv(str(file_path), header=None, names=[name], delimiter=delimiter).fillna(0)
    except Exception as e:
        raise Exception(f"Error reading .txt file: {e}")


def etl_excel_file(relative_path, sheet_name):
    """
    Load data from an Excel file (.xls).

    Extract: Load .xls data

    Transform:
        1. Data passed do not have header, so do not use header when reading the file
        2. Use delimiter passed as param to separate values
        3. Transpose data, because Excel has 256 column number restriction and input data provide columns as rows

    Load: Return DataFrame containing the data

    :param relative_path: Relative path to the .xls file.
    :param sheet_name: Name or index of the sheet to read.
    :return: DataFrame containing the data.
    """
    global file_path
    try:
        # Get the absolute path to the file
        file_path = get_absolute_path(relative_path)
        # Start reading the file
        # Use header none, so it does not waste first line of data as header
        # Use delimiter passed as parameter
        # Transpose, due to Excel column limitation
        # return DataFrame containing data
        return pandas.read_excel(file_path, header=None, sheet_name=sheet_name).transpose().fillna(0)
    except Exception as e:
        raise Exception(f"Error reading .xls file: {e}")


def etl_accdb_file(relative_path, table_name):
    """
    Load data from an Access database file (.accdb)

    Extract: Load .accdb data, after creating a connection with database

    Transform:
        1. Change headers from "Πεδίο1..Πεδίο150" to "0..149", in order to match with other data
        2. Transpose data, because Access database has 255 column number restriction
        and input data provide columns as rows

    Load: Return DataFrame containing the data

    :param relative_path: Relative path to the Access database file from the project root directory.
    :return: DataFrame containing the data.
    """
    global file_path
    conn = None
    try:
        # Get the absolute file path to the Access database file
        file_path = get_absolute_path(relative_path)

        # Construct connection string for Access
        conn_str = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={file_path};"

        # Establish connection
        conn = pyodbc.connect(conn_str)

        # Construct query
        query = f"SELECT * FROM {table_name};"

        # Execute query given Access db connection
        df = pandas.read_sql(query, conn)

        # Rename columns using regular expression to extract numeric part
        df.columns = [re.search(r'\d+', col).group() for col in df.columns]

        # Transpose, due to Access database column limitation
        # return DataFrame containing data
        return pandas.DataFrame(df.transpose().fillna(0))
    except Exception as e:
        raise Exception(f"Error reading .accdb file: {e}")
    finally:
        conn.close()
