import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage

from skimage import measure, morphology

import random
import datetime
import logging
import progressbar
import argparse
import shutil

random.seed(a=42)

MIN_BOUND = -1000.0
MAX_BOUND = 400.0

# from https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def show_shapes(image, scan):
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    logging.info("spacing:\t\t\t{0}".format(spacing))
    resize_factor = spacing / [1,1,1]
    logging.info("resize_factor:\t\t\t{0}".format(resize_factor))
    logging.info("image.shape:\t\t\t{0}".format(image.shape))
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    logging.info("new_shape:\t\t\t{0}".format(new_shape))
    real_resize_factor = new_shape / image.shape
    logging.info("real_resize_factor:\t\t{0}".format(real_resize_factor))
    new_spacing = spacing / real_resize_factor
    logging.info("new_spacing:\t\t\t{0}".format(new_spacing))


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


def crop(image):
    image[image>MAX_BOUND] = MAX_BOUND
    image[image<MIN_BOUND] = MIN_BOUND
    return image


def shift_to_center(image, background=0):
    center = np.array(scipy.ndimage.measurements.center_of_mass(image != background)) + 1
    shape_size = np.array(image.shape)
    shift = np.round(np.ceil((shape_size / 2.)) - center).astype(np.int)
    if np.any(shift > 0):
        result = scipy.ndimage.interpolation.shift(image, shift).astype(image.dtype)
        return result, shift
    return image, None


def change_paddings(image, new_size=(400, 400, 400)):
    new_size = np.array(new_size)
    image_size = np.array(image.shape)
    assert len(image_size) == len(new_size)

    def zip_it(p):
        return zip(np.ceil(p).astype(np.int), np.floor(p).astype(np.int))

    pad = new_size - image_size
    pad = pad / 2.
    pad_add, pad_sub = pad.copy(), pad.copy()
    pad_add[pad_add < 0] = 0
    pad_sub[pad_sub > 0] = 0
    pad_sub *= -1
    pad_add = zip_it(pad_add)
    pad_sub = zip_it(pad_sub)

    image = np.lib.pad(image, pad_add, mode='constant', constant_values=0)

    for i, ps in enumerate(pad_sub):
        del_arr = []
        if (ps[0] > 0):
            del_arr = range(image_size[i])[:ps[0]]
        if (ps[1] > 0):
            del_arr += range(image_size[i])[-ps[1]:]
        image = np.delete(image, del_arr, axis=i)

    assert np.array_equal(np.array(image.shape), new_size)
    return image


def convert_dicom(dicom_path, width=300, height=300, length=300, is_half_reduce=False):
    patient = load_scan(dicom_path)
    patient_pixels = get_pixels_hu(patient)
    show_shapes(patient_pixels, patient)
    result, spacing = resample(patient_pixels, patient, [1.25, 1, 1.25])

    # create mask
    segmented_lungs_fill = segment_lung_mask(result, True)
    segmented_lungs_fill_dilated = scipy.ndimage.binary_dilation(segmented_lungs_fill, iterations=10)

    result = crop(result * segmented_lungs_fill_dilated)
    logging.info("result size:\t{0}".format(result.shape))

    result = change_paddings(result, new_size=(height, length, width))
    logging.info("size after add paddings:\t{0}".format(result.shape))

    result, shift = shift_to_center(result)
    if shift is not None:
        logging.info("shift: {0}".format(shift))

    if is_half_reduce:
        result = measure.block_reduce(result, block_size=(2, 2, 2), func=np.max)
    return result


def convert_dicoms(dicom_path, labels_file, result_folder=None, log_file=None,
                   batch_size=3, total_proc_number=1, current_proc_number=1,
                   is_delete_old_files=False, width=300, height=300, length=300, is_half_reduce=False):
    labels = pd.read_csv(labels_file)
    result_def_name = "dicom_" + datetime.datetime.now().strftime('%y_%m_%d')

    if result_folder is None:
        result_folder = os.path.join(dicom_path, (result_def_name))
        print("Result folder: {0}".format(result_folder))
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    result_folder_submission = os.path.join(result_folder, 'submission/')
    if not os.path.exists(result_folder_submission):
        os.makedirs(result_folder_submission)

    if log_file is None:
        log_folder = result_folder or dicom_path
        log_file = os.path.join(
            log_folder,
            (result_def_name + "_" + str(total_proc_number) + "_" + str(current_proc_number) + ".log"))
        print("Log file: {0}".format(log_file))
    if not os.path.exists(os.path.dirname(os.path.abspath(log_file))):
        os.makedirs(os.path.dirname(os.path.abspath(log_file)))

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s')

    patients = os.listdir(dicom_path)
    random.shuffle(patients)
    patients, filename_count = crop_path(
        patients,
        total_proc_number=total_proc_number,
        current_proc_number=current_proc_number,
        batch_size=batch_size)

    total_saved = 0
    total_count = 0
    total = len(patients)

    ds_main = DataSaver(batch_size=batch_size, save_path=result_folder, filename_count=filename_count)
    ds_subm = DataSaverSubmission(save_path=result_folder_submission)

    bar = progressbar.ProgressBar(
        maxval=total,
        widgets=[
            "Preprocessing: ",
            " ", progressbar.SimpleProgress(),
            " / ", progressbar.Timer(),
            " / ", progressbar.ETA()
        ]).start()

    def get_label(id):
        try:
            res = labels[labels["id"] == id].cancer.values[0].astype(np.bool)
            return np.array([res])
        except Exception:
            return None

    # for patient in patients:
    for patient_name in patients:
        try:
            start = datetime.datetime.now()
            logging.info("Patient {0}...".format(patient_name))
            label = get_label(patient_name)

            # read slices
            path = os.path.join(dicom_path, patient_name)
            result = convert_dicom(path, width=width, height=height, length=length, is_half_reduce=is_half_reduce)

            if label is not None:
                ds_main.save(label=label, image=result)
            else:
                ds_subm.save(patient_name=patient_name, image=result)
                logging.info("it's submission: {0}".format(patient_name))

            if is_delete_old_files:
                logging.info("remove {0}".format(path))
                shutil.rmtree(path)

            total_saved += 1
            logging.info("done, {0}\n".format(str(datetime.datetime.now() - start)))
        except:
            logging.exception("Error with {0}".format(patient_name))
        finally:
            total_count += 1
            bar.update(total_count)
    ds_main.close()
    bar.finish()
    print("Done. {0} files saved.".format(total_saved))


class DataSaver:
    def __init__(self, batch_size, save_path, filename_count):
        self.batch_size = batch_size
        self.save_path = save_path
        self.cash = []
        self.filename_count = filename_count

    def get_filename(self):
        return os.path.join(self.save_path, str(self.filename_count).zfill(4) + ".bin")

    def save(self, label, image):
        self.cash.append((
            label.tobytes() if label is not None else np.array([False]),
            image.tobytes()))
        if len(self.cash) >= self.batch_size:
            self._save()
        logging.info("save to {0}".format(self.get_filename()))

    def _save(self):
        with open(self.get_filename(), 'ab+') as f:
            for c in self.cash:
                f.write(c[0])
                f.write(c[1])
        self.filename_count += 1
        self.cash = []

    def close(self):
        self._save()


class DataSaverSubmission:
    def __init__(self, save_path):
        self.save_path = save_path

    def get_filename(self, patient_name):
        return os.path.join(self.save_path, patient_name + ".bin")

    def save(self, patient_name, image):
        with open(self.get_filename(patient_name), 'ab+') as f:
            f.write(image.tobytes())
        logging.info("save to {0}".format(self.get_filename(patient_name)))


def crop_path(paths, total_proc_number, current_proc_number, batch_size):
    current_proc_number -= 1
    total_len = len(paths)
    skip = total_len / float(total_proc_number) * current_proc_number
    skip = int(round(skip / batch_size) * batch_size)

    up_to = total_len / float(total_proc_number) * (current_proc_number + 1)
    up_to = int(round(up_to / batch_size) * batch_size)

    return paths[skip:up_to], (skip / batch_size)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dicom_path', '-d', dest='dicom_path', required=True)
    parser.add_argument('--labels_file', '-l', dest='labels_file', required=True)
    parser.add_argument('--result_file', '-r', dest='result_file')
    parser.add_argument('--log_file', '--log', dest='log_file')
    parser.add_argument('--batch_size', '--bs', dest='batch_size', default=3)
    parser.add_argument('--total_proc_number', '--tpn', dest='total_proc_number', default=1)
    parser.add_argument('--current_proc_number', '--cpn', dest='current_proc_number', default=1)
    parser.add_argument('--is_delete_old_files', '--df', dest='is_delete_old_files', default=0)
    parser.add_argument('--width', '--wd', dest='width', default=300)
    parser.add_argument('--height', '--he', dest='height', default=300)
    parser.add_argument('--length', '--ln', dest='length', default=300)
    parser.add_argument('--is_half_reduce', '--hr', dest='is_half_reduce', default=0)
    args = parser.parse_args()
    convert_dicoms(
        args.dicom_path, args.labels_file,
        args.result_file, args.log_file,
        batch_size=int(args.batch_size),
        total_proc_number=int(args.total_proc_number),
        current_proc_number=int(args.current_proc_number),
        width=int(args.width),
        height=int(args.height),
        length=int(args.length),
        is_half_reduce=(int(args.is_half_reduce) == 1),
        is_delete_old_files=(int(args.is_delete_old_files) == 1))


if __name__ == '__main__':
    main()
