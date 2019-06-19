import os
import cv2
import math
import imutils
import numpy as np

def get_six_lane_labels(_line):
    data_raw = []
    coe1 = []
    coe2 = []

    _line = _line.strip('\r\n')
    _line += ' '
    line_content = _line.split(' ')

    for i in range(len(line_content) - 1):
        data_raw.append(line_content[i])

    for i in range(0, 3):
        coe1.append(data_raw[i])

    for i in range(3, 6):
        coe2.append(data_raw[i])

    return coe1, coe2

def get_lane_labels(_line):
    data_raw = []
    contains_lane = False
    lane_prediction = []
    coe1 = []
    coe2 = []
    coe3 = []
    coe4 = []

    _line = _line.strip('\r\n')
    _line += ' '
    line_content = _line.split(' ')

    for i in range(len(line_content) - 1):
        data_raw.append(line_content[i])

    for i in range(4):
        # print(i)
        if data_raw[i] == '1':
            contains_lane = True
            break
        else:
            contains_lane = False

    for i in range(4):
        lane_prediction.append(data_raw[i])

    for i in range(4, 7):
        coe1.append(data_raw[i])

    for i in range(7, 10):
        coe2.append(data_raw[i])

    for i in range(10, 13):
        coe3.append(data_raw[i])

    for i in range(13, -1):
        coe4.append(data_raw[i])

    return contains_lane, lane_prediction, coe1, coe2, coe3, coe4


def get_mask_coordinates(mask_image, channel='B', val=100):
    if channel == 'B':
        cval = 0
    elif channel == 'G':
        cval = 1
    elif channel == 'R':
        cval = 2
    else:
        assert False, "cval not set"

    coord = np.argwhere(mask_image[:, :, cval] > val)
    # print(coord)

    _y = coord[:, 0]
    _x = coord[:, 1]

    # print('coord:', coord)
    #
    # print('x:', _x)
    # print('y:', _y)

    return _x, _y


def get_mask_coordinates_v2(mask_image, channel_val=()):
    # coordb = np.argwhere((mask_image[:, :, 0] == channel_val[0]) and (mask_image[:, :, 1] == channel_val[1]) and
    #                     (mask_image[:, :, 2] == channel_val[2]))

    # coord = np.where((mask_image == channel_val))

    # print(coord[1])

    _x = []
    _y = []

    if (channel_val[0] - 1) < 0:
        lw_b = 0
    else:
        lw_b = channel_val[0] - 1

    if (channel_val[1] - 1) < 0:
        lw_g = 0
    else:
        lw_g = channel_val[0] - 1

    if (channel_val[2] - 1) < 0:
        lw_r = 0
    else:
        lw_r = channel_val[0] - 1

    lower_tresh = np.array([lw_b, lw_g, lw_r])
    upper_tresh = np.array([channel_val[0], channel_val[1], channel_val[2]])

    print(lower_tresh)
    mask = cv2.inRange(mask_image, lower_tresh, upper_tresh)

    coord = cv2.findNonZero(mask)
    print(coord)
    print(len(coord))

    for i in range(len(coord)):
        _x.append(coord[i][0][0])
        _y.append(coord[i][0][1])

    # print(coord[0][0][0])
    # print(coord[0])
    # print(list(zip(*coord[:, :, 0])[0]))
    # _y = list(coord[:, :, 0])
    # _x = list(coord[:, :, 1])

    print('x:', _x)
    print('y:', _y)

    return _x, _y


def draw_from_coords(_img, _x, _y, color=(255, 0, 255)):
    print("pppp", _x[6])
    if len(_x) == len(_y):
        for j in range(len(_x)):
            # circle(img, center, radius, color, thickness=None, lineType=None, shift=None)
            cv2.circle(_img, (_x[j], _y[j]), 1, color, thickness=-1, lineType=1, shift=0)
            # print(_x[j], _y[j])

        cv2.imshow("view", _img)
    return _img


def draw_from_coefficient(_img, _coe, _height, color=(0, 0, 0)):
    """
    :param color:
    :param _img: image to draw
    :param _coe: pass the 2 deg coefficient as list.
    ex: _coe = [-1.0000000e-04 -2.1454000e+00  1.4350935e+03]
    :param _height: the width of the raw image
    :return: nothing
    """
    for i in range(_height):
        a = float(_coe[0])
        b = float(_coe[1])
        c = float(_coe[2]) - i

        if a == 0:
            a = 0.0001

        if (b ** 2 - (4 * a * c)) < 0: continue

        d = math.sqrt((b ** 2) - (4 * a * c))
        x = (-b - d) // (2 * a)
        # print('det: ', -b - d, 'x_hat:', x, ',y_hat:', i)
        # print(color)
        cv2.circle(_img, (i, int(x)), 1, color, thickness=-1, lineType=8, shift=0)
        x = (-b + d) // (2 * a)
        cv2.circle(_img, (i, int(x)), 1, color, thickness=-1, lineType=8, shift=0)

    return _img


def get_crop_params(_org_width, _org_height, _crop_percentage=0.1):
    i_row_list = []
    f_row_list = []
    i_column_list = []
    f_column_list = []

    # 2, 4, 7, 8,

    # roi = flipped_img[:100, :100, :]

    # 1
    i_row_list.append(0)
    f_row_list.append(int(_org_width - (_org_width * _crop_percentage)))

    i_column_list.append(0)
    f_column_list.append(int(_org_height))

    # 2
    i_row_list.append(0)
    f_row_list.append(int(_org_width))

    i_column_list.append(int(_org_height * _crop_percentage))
    f_column_list.append(int(_org_height))

    # 3
    # i_row_list.append(0)
    # f_row_list.append(int(_org_width))
    #
    # i_column_list.append(0)
    # f_column_list.append(int(_org_height - (_org_height * _crop_percentage)))

    # 4
    i_row_list.append(int(_org_width * _crop_percentage))
    f_row_list.append(int(_org_width))

    i_column_list.append(0)
    f_column_list.append(int(_org_height))

    # 5
    # i_row_list.append(int(_org_width * _crop_percentage))
    # f_row_list.append(int(_org_width))
    #
    # i_column_list.append(0)
    # f_column_list.append(int(_org_height - (_org_height * _crop_percentage)))

    # 6
    i_row_list.append(int(_org_width * _crop_percentage))
    f_row_list.append(int(_org_width))

    i_column_list.append(int(_org_height * _crop_percentage))
    f_column_list.append(int(_org_height))

    # 7
    i_row_list.append(int(_org_width * _crop_percentage))
    f_row_list.append(int(_org_width - (_org_width * _crop_percentage)))

    i_column_list.append(0)
    f_column_list.append(int(_org_height))

    # 8
    # i_row_list.append(0)
    # f_row_list.append(int(_org_width - (_org_width * _crop_percentage)))
    #
    # i_column_list.append(0)
    # f_column_list.append(int(_org_height - (_org_height * _crop_percentage)))

    # 9
    # i_row_list.append(0)
    # f_row_list.append(int(_org_width))
    #
    # i_column_list.append(int(_org_height * _crop_percentage))
    # f_column_list.append(int(_org_height - (_org_height * _crop_percentage)))

    # 10
    i_row_list.append(0)
    f_row_list.append(int(_org_width - (_org_width * _crop_percentage)))

    i_column_list.append(int(_org_height * _crop_percentage))
    f_column_list.append(int(_org_height))

    return i_row_list, f_row_list, i_column_list, f_column_list


def do_rotate_images(_img):
    return


if __name__ == "__main__":
    a, b, c, d = get_crop_params(1000, 1000, _crop_percentage=0.2)
    image = cv2.imread("./_dataset/05312327_0001-05340.jpg")

    # loop over the rotation angles again, this time ensuring
    # no part of the image is cut off
    for angle in np.arange(0, 5, 1):
        rotated = imutils.rotate(image, angle)
        cv2.imshow("Rotated (Correct)", rotated)
        cv2.waitKey(0)

    for angle in np.arange(355, 360, 1):
        rotated = imutils.rotate(image, angle)
        cv2.imshow("Rotated (Correct)", rotated)
        cv2.waitKey(0)

        # print()


def noisy(noise_typ, image):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    :param image: mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.

    :param noise_typ:
    """
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.01
        sigma = var ** 0.6
        gauss = np.random.normal(mean, 1, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        noisy = noisy.astype('uint8')
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.3
        amount = 0.01
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        noisy = noisy.astype('uint8')
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        noisy = noisy.astype('uint8')
        return noisy

