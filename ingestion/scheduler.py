"""
ingestion/scheduler.py
----------------------
PREVIEW!!!
Entry point for scheduling the updates of the securities prices.
"""


def entry_method(tuple):
    # args: (security, date_range, collector) tuple for parallel storing
    # parallel calling update_security for all securities in tuple

    pass


def update_security(security, date_range, collector: str = "ArivaScraper"):
    # fetch data
    # call normalizer to normalize data, normalizer returns data as OHLCVRow objects
    # store OHLCVRow object

    pass
