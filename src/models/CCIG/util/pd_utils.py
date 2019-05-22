# coding:utf-8
import csv
import pandas as pd


def export_columns(fin, fout, col_list,
                   sep_in, sep_out, keep_header=False):
    """
    Export column(s) from a file.
    """
    df = pd.read_csv(fin, sep=sep_in)
    df_out = df[col_list]
    df_out.to_csv(fout, sep=sep_out, header=keep_header, index=False)


def import_column(fin, fcol, fout, col,
                  sep_in, sep_out, contain_header=False):
    """
    Merge a column from a file.
    """
    fcol = open(fcol, "r")
    lines = fcol.read().splitlines()
    df = pd.read_csv(fin, sep=sep_in, quoting=csv.QUOTE_NONE)
    df[col] = lines
    df.to_csv(fout, sep=sep_out, header=True, index=False)
