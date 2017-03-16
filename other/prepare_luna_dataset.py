import os
import shutil
import argparse
import pandas as pd
import logging

RESULT_FOLDER_NAME = "prepared_dicoms"

def is_contains_dicom(root_path):
    if os.path.isdir(root_path):
        count = 0
        for fname in os.listdir(root_path):
            path = os.path.join(root_path, fname)
            if os.path.isfile(path) and path.endswith(".dcm"):
                count += 1
        if count > 50:
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
    parser.add_argument('--annotations', '-a', dest='annotations', required=True)
    args = parser.parse_args()
    dicom_path = args.dicom_path
    annotations = args.annotations

    if not os.path.isdir(dicom_path):
        raise ValueError('Wrong dataset directory {0}'.format(dicom_path))

    if not os.path.isfile(annotations) and not annotations.endswith(".csv"):
        raise ValueError('Wrong annotations directory {0}'.format(annotations))

    dicom_path_list = os.listdir(dicom_path)

    result_path = os.path.join(dicom_path, RESULT_FOLDER_NAME)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(result_path, 'prepare_time_dataset.log'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info("id,cancer")

    annotations = pd.read_csv(annotations, index_col=0)

    for fname in dicom_path_list:
        path = os.path.join(dicom_path, fname)

        dicom_folders = get_dicom_folders(path)
        for df in dicom_folders:
            dicom_label = os.path.basename(os.path.normpath(df))
            is_have_nodule = dicom_label in annotations.index
            is_have_nodule = 1 if is_have_nodule else 0

            new_path = os.path.join(result_path, dicom_label)
            if not os.path.exists(new_path):
                shutil.move(df, new_path)
                logger.info("{0},{1}".format(dicom_label, is_have_nodule))

    print("done")

if __name__ == '__main__':
    main()
