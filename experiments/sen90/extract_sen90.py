import sys
import pandas as pd
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
except ImportError:
    print("rpy2 is required.")
    sys.exit(1)

try:
    pscl = importr("pscl")
    wnominate = importr("wnominate")
    
    robjects.r('''
    data(sen90)
    votes <- sen90$votes
    legis_data <- sen90$legis.data
    
    votes_df <- as.data.frame(votes)
    legis_df <- as.data.frame(legis_data)
    legis_df$id <- rownames(legis_df)
    ''')
    
    with localconverter(robjects.default_converter + pandas2ri.converter):
        votes_df = robjects.conversion.rpy2py(robjects.globalenv['votes_df'])
        legis_df = robjects.conversion.rpy2py(robjects.globalenv['legis_df'])
    
    # Save to disk
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    votes_df.to_csv(os.path.join(data_dir, "sen90_votes.csv"), index=False)
    legis_df.to_csv(os.path.join(data_dir, "sen90_legis.csv"), index=False)
    print("Successfully extracted sen90 data to CSVs.")
except Exception as e:
    print(f"Error extracting data: {e}")
