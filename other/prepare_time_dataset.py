import os
import shutil
import argparse

RESULT_FOLDER_NAME = "prepared_dicoms"

def is_contains_dicom(root_path):
    if os.path.isdir(root_path):
        for fname in os.listdir(root_path):
            path = os.path.join(root_path, fname)
            if os.path.isfile(path) and path.endswith(".dcm"):
                return True
    return False


def get_dicom_folders(root_path):
    if os.path.isfile(root_path):
        return None
    if is_contains_dicom(root_path):
        return [root_path]

    result = []
    for fname in os.listdir(root_path):
        path = os.path.join(root_path, fname)
        r = get_dicom_folders(path)
        if r is not None:
            result.extend(r)
    return result

def main():
    print("start")
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dicom_path', '-d', dest='dicom_path', required=True)
    args = parser.parse_args()
    dicom_path = args.dicom_path

    if dicom_path is None:
        raise ValueError('You must supply the {0}'.format("dicom_path"))
    if not os.path.isdir(dicom_path):
        raise ValueError('Wrong dataset directory {0}'.format("dicom_path"))

    result_path = os.path.join(dicom_path, RESULT_FOLDER_NAME)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for fname in os.listdir(dicom_path):
        path = os.path.join(dicom_path, fname)
        pathxml = path + ".xml"

        dicom_folders = get_dicom_folders(path)
        if dicom_folders is not None and len(dicom_folders) > 0 and os.path.isfile(pathxml):
            is_have_nodule = 'noduleID' in open(pathxml).read()
            is_have_nodule = 1 if is_have_nodule else 0
            i = 0
            for df in dicom_folders:
                sample_name = "{0}_{1}".format(fname, i)
                new_path = os.path.join(result_path, sample_name)
                shutil.move(df, new_path)
                i += 1
                print("{0};{1}".format(sample_name, is_have_nodule))

    print("done")

if __name__ == '__main__':
    main()
