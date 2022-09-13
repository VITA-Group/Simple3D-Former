import argparse
import os
import subprocess
import sys

"""
From: https://raw.githubusercontent.com/Ryanglambert/3d_model_retriever/master/binvox_converter.py
To create binvox of ModelNet
"""

if 'linux' in sys.platform:
    # CMD = ("xvfb-run -s '-screen 0 640x480x24'" 
    #        "./binvox -cb -pb -e -c -d {dim} {path}")
    CMD = "../utils/binvox -cb -pb -e -c -d {dim} {path}"
elif 'darwin' in sys.platform:
    CMD = '../utils/binvox -cb -e -c -d {dim} {path}'
else:
    raise SystemError('System is not of type darwin or linux, what kind of system are you using?')


def get_rid_of_misplaced_text(path):
    with open(path, 'r') as f:
        text = f.read()

    if text[:3] == 'OFF':
        new_text = 'OFF\n' + text[3:]
        with open(path, 'w') as f:
            f.write(new_text)


def convert_off_to_binvox(path, dim=30):
    "Converts *.off file to *.binvox"
    if ".off" not in path:
        print("{} is not an '*.off' file... skipping".format(path))
        return

    get_rid_of_misplaced_text(path)

    print('CONVERTING {}'.format(path))
    process = subprocess.Popen(CMD.format(dim=dim, path=path).split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print("Output from Binvox: \n{}".format(output))
        raise Exception("Binvox didn't do something right\nBinvox Error:\n{}".format(error))


def list_file_paths(path):
    ignore_files = ['.DS_Store']
    for root, _, files in os.walk(path):
        for file_path in files:
            # this is throwing a funky error but the files are still showing up will come back to this when it's not crunchtime
            # /Users/ryan/repos/deep_learning_mabs/ModelNet10/sofa/train/sofa_0364.off
            # wc: wc: open: No such file or directory
            # wc: -l: open: No such file or directory
            if file_path not in ignore_files:
                abs_path = os.path.abspath(os.path.join(root, file_path))
                print(abs_path)
                yield abs_path


def _remove_all(path):
    paths = list_file_paths(path)
    for file_path in paths:
        if ".binvox" in file_path:
            print("removing...{}".format(file_path))
            os.remove(file_path)


def main():
    parser = argparse.ArgumentParser(description='Process *.off files into *.binvox')
    parser.add_argument(dest="root_path",
                        help="give the root_path for the *.off files you want to convert to *.binvox")
    parser.add_argument('--dimensions', default=30)
    parser.add_argument('--remove-all-dupes', dest='remove_all', action='store_true')
    args = parser.parse_args()

    # assert 'ModelNet' in args.root_path, ("ONLY RUN THIS IN THE ModelNet folder!!!")
    # Essentially to overwrite what's already there
    if args.remove_all:
        _remove_all(args.root_path)

    # actually do the conversion from *.off to *.binvox
    paths_generator = list_file_paths(args.root_path)
    for path in paths_generator:
        convert_off_to_binvox(path, dim=args.dimensions)


if __name__ == '__main__':
    main()