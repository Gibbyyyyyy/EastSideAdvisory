

import os
from pathlib import Path

def openSheet(location, sheetNumber):

    dataPathBase = Path(location) # folder where the sheets are located
    # actually, its the specific folder that may or may not be a few directories down from dataPathBase
    target_folder = next((p for p in dataPathBase.rglob(str(sheetNumber)) if p.is_dir()), None)

    if not target_folder:
        raise FileNotFoundError(f"No folder named '{sheetNumber}' found under {dataPathBase}") # oops not in there

    # Get all files in this folder, as path objects (pathlib)
    samplePathObjects = [f for f in target_folder.rglob('*') if f.is_file() and "detail" in f.name.lower()] 

    return samplePathObjects # now I know the particular locations of all files I care about, as well as names of files and other attributes
    
# debug code
if __name__ == "__main__":
    
    sheetNum = 41
    samplePathObjects = openSheet(r"C:\EastSideAdvisory\EE416kalmia\Python\Lab Data",sheetNum)

  