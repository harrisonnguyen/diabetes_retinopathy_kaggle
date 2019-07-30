import os
import click
from PIL import Image
from shutil import copyfile



@click.command()
@click.argument('dir-path',
    type=click.Path(file_okay=False, dir_okay=True,exists=True))
@click.argument('to-path',
    type=click.Path(file_okay=False, dir_okay=True))
def main(dir_path,to_path):
    """This script converts tiff files to jpeg in a given directory"""
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            if os.path.splitext(os.path.join(root, name))[1].lower() == '.jpeg' or os.path.splitext(os.path.join(root, name))[1].lower() =='.jpg':
                print("jpeg exists for {}".format(name))
                #copyfile(os.path.join(root, name), os.path.join(to_path, name))
            elif os.path.splitext(os.path.join(root, name))[1].lower() == ".tiff" or os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
                outfile = os.path.splitext(os.path.join(to_path, name))[0] + ".jpeg"
                if not os.path.exists(outfile):
                    try:
                        im = Image.open(os.path.join(root, name))
                        print("Generating jpeg for {}".format(name))
                        im.thumbnail(im.size)
                        im.save(outfile, "JPEG", quality=100)
                    except Exception as e:
                        print(e)

if __name__ == "__main__":
    exit(main())  # pragma: no cover
