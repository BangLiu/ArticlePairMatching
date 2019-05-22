# coding=utf-8
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.getcwd()
sys.path.insert(0, parent_dir)
sys.path.append('/usr/local/Cellar/graph-tool/2.27_5/lib/python3.7/site-packages/')  # for load graph-tool
