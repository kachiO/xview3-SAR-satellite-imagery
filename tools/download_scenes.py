"""Download xView3 SAR Scenes."""
import subprocess
import argparse
from pathlib import Path


def parse_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description="Download files.")

    parser.add_argument( "--input-dir", type=str, default='/opt/ml/processing/input/',
                        help="Path to folder containing files.",)
    parser.add_argument( "--output-dir", type=str, default='/opt/ml/processing/output/',
                        help="Folder to download files.",)
    parser.add_argument('--prefix', type=str, default='', 
                        help='Prefix used for download list. Default None.' )
    parser.add_argument('--debug', '-d', default=False, type=bool, 
                        help='Debug mode. Set aria2c')
    return parser.parse_args()

def main(args):
    """Run main function. 
    
    Download files in .txt file. Loop through all files in folder.
    """
    
    prefix = args.prefix
    output_dir = Path(args.output_dir)
    
    for file in Path(args.input_dir).glob(f'*{prefix}'):
            
        process = subprocess.Popen(['aria2c', f'--input-file={str(file)}', 
                                    '--auto-file-renaming=false', '--continue=true', 
                                    f'--dir={str(output_dir)}', f'--dry-run={str(args.debug).lower()}'])
        process.wait()

if __name__ == "__main__":
    args = parse_args()
    main(args)