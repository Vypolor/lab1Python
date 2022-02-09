import numpy as np
import matplotlib.pyplot as plt
import argparse

from skimage import io


def validate_k(k):
    if k < 0:
        raise ValueError("k value should be in range (0,1)")
    elif k > 1:
        raise ValueError("k value should be in range (0,1)")


def multiply_rule(first_img, second_img, k):
    calculated_first_img = first_img * k
    k_for_second_img = 1 - k
    calculated_second_img = second_img * k_for_second_img
    result = np.multiply(calculated_first_img, calculated_second_img)
    return result


# def resize_smallest_image(image_to_change, new_shape):
#     changed_arr = np.zeros(new_shape, dtype=int)
#     changed_arr[0:image_to_change.shape[0], 0:image_to_change.shape[1], 0:image_to_change.shape[2]] = image_to_change
#     return changed_arr


def resize_biggest_image(image_to_change, new_shape):
    size = new_shape[0] * new_shape[1] * new_shape[2]
    old_size = image_to_change.shape[0] * image_to_change.shape[1] * image_to_change.shape[2]
    image_to_change = image_to_change.reshape(old_size)
    changed_arr = image_to_change[0:size]
    changed_arr = changed_arr.reshape(new_shape)
    return changed_arr


def pars_input_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('-image1', '--first_picture', type=str, help='Absolute path to first image', required=True)
    parser.add_argument('-image2', '--second_picture', type=str, help='Absolute path to second image', required=True)
    parser.add_argument('-k', '--k_value', type=float, help='multiple exposure ratio'
                        , required=True)
    #parser.add_argument('-pr', '--path_to_result_image', type=str, help='Absolute path to result image', required=False)
    args = parser.parse_args()
    return args


parameters = pars_input_parameters()
first_img = io.imread(parameters.first_picture)
second_img = io.imread(parameters.second_picture)
k = parameters.k_value

#first_img = io.imread("C:/Users/nifr0821/Desktop/test/test.jpg")
#second_img = io.imread("C:/Users/nifr0821/Desktop/test/test2.jpg")
#k = 0.9
validate_k(k)
first_arr = np.array(first_img, np.uint8)
second_arr = np.array(second_img, np.uint8)
# --------------------------------Resize smallest image-----------------------------------
# if first_arr.shape > second_arr.shape:
#     second_arr = resize_smallest_image(second_arr, first_arr.shape)
# elif second_arr.shape > first_arr.shape:
#     first_arr = resize_smallest_image(first_arr, second_arr.shape)


# --------------------------------Resize biggest image-----------------------------------
if first_arr.shape > second_arr.shape:
    first_arr = resize_biggest_image(first_arr, second_arr.shape)
elif second_arr.shape > first_arr.shape:
    second_arr = resize_biggest_image(second_arr, first_arr.shape)

multiply_rule(first_arr, second_arr, k)
result_arr = np.multiply(first_arr, second_arr)

fig, axes = plt.subplots(1, 3, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(first_img)
ax[0].set_title("First Image")
ax[1].imshow(second_img)
ax[1].set_title("Second Image")
ax[2].imshow(result_arr)
ax[2].set_title("Result")

fig.tight_layout()
plt.show()

# ---------------- Save result image ------------------------
io.imsave("result.jpg", result_arr)
fig.savefig("result2.jpg")
