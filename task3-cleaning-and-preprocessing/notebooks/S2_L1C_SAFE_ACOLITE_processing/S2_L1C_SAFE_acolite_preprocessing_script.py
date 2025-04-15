# run_acolite_single_safe.py
import sys
import os
# Clone the ACOLITE repository : git clone https://github.com/acolite/acolite.git
# Add ACOLITE root to sys.path
# The path to use here is the one that points to the acolite/ folder 
# containing LICENCE.txt
sys.path.append('your-path-to-acolite-repo')

# Import acolite_run
from acolite.acolite.acolite_run import acolite_run

# Define settings
settings = {
    'inputfile': 'C:\\omdena-projects\\plastic2025\\Data\\S2B_MSIL1C_20250409T104619_N0511_R051_T31UDT_20250409T113539.SAFE',  # Replace with your .SAFE path
    'output': './acolite_output',
    'l2r_export_geotiff': True,  # Ensure TIFF export
    'verbosity': 5,  # Detailed logging
    # Optional: Add other settings as needed
    's2_target_res': 10,  # 10m resolution
    'atmospheric_correction_method': 'dark_spectrum'
}

# Ensure output directory exists
output_dir = settings['output']
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Run ACOLITE
print(f"Processing {settings['inputfile']}...")
output_files = acolite_run(settings=settings, inputfile=settings['inputfile'], output=settings['output'])

# Print results
print("Output files:", output_files)