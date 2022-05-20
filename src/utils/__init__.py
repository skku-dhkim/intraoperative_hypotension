from pandas import DataFrame
import os

cpu_counts = os.cpu_count()
__all__ = [
    'DataFrame',
    'cpu_counts'
]