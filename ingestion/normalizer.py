"""
ingestion/normalizer.py
-----------------------
PREVIEW!!!
"""


# single securities or list of securities?
# so is the normalizer the entry point for fetching and storing data? Should the entry point somewhere different?
# Maybe add a new scheduler.py file as the entry point and scheduling the updates + parallel is possible?
# should the scheduler than 1) fetch data from collector, 2) normalize it and 3) store it via the repository?
def to_do_method(date_range, collector: str = "ArivaScraper"):
    if collector == "ArivaScraper":
        # fetch data
        # change data into OHLCVRow
        # store Rows in db
        pass

    # other collector's
