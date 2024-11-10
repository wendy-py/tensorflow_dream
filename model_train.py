import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

Sample_Image = tf.keras.preprocessing.image.load_img('data/img_0.jpg')
print(f"shape: {np.shape(Sample_Image)}")
print(f"type: {type(Sample_Image)}")

Sample_Image = tf.keras.preprocessing.image.img_to_array(Sample_Image)
print(f"type after img_to_array: {type(Sample_Image)}")
print(f'min pixel values = {Sample_Image.min()}, max pixel values = {Sample_Image.max()}')

Sample_Image = np.array(Sample_Image)/255.0
print(f"shape after normalise: {Sample_Image.shape}")
print(f'min pixel values = {Sample_Image.min()}, max pixel values = {Sample_Image.max()}')

Sample_Image = tf.expand_dims(Sample_Image, axis = 0)
print(f"shape after expand_dims: {np.shape(Sample_Image)}")

plt.imshow(np.squeeze(Sample_Image))
plt.show()

base_model = tf.keras.applications.InceptionV3(include_top = False, weights = 'imagenet')
print(base_model.summary());

names = ['mixed3', 'mixed5', 'mixed7']
layers = [base_model.get_layer(name).output for name in names]

# Create the feature extraction model
deepdream_model = tf.keras.Model(inputs = base_model.input, outputs = layers)
print(deepdream_model.summary())

# Let's run the model by feeding in our input image and taking a look at the activations "Neuron outputs"
activations = deepdream_model(Sample_Image)
print(activations)
print(len(activations))

# - tf.GradientTape() is used to record operations for automatic differentiation
# - For example, Let's assume we have the following functions y = x^3.
# - The gradient at x = 2 can be computed as follows: dy_dx = 3 * x^2 = 3 * 2^2 = 12.
x = tf.constant(2.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = x * x * x
dy_dx = g.gradient(y, x) # Will compute to 12
print(f"check calc, dy_dx (should be 12.0): {dy_dx}")

# - CREDITS: The DeepDream Code has been adopted from Keras Documentation:
# - https://www.tensorflow.org/tutorials/generative/deepdream
# Since the calc_loss function includes expand dimension, let's squeeze the image (reduce_dims)
Sample_Image = tf.squeeze(Sample_Image)

# IMPLEMENT DEEP DREAM ALGORITHM - STEP #1 (CALCULATE THE LOSS)
# Function used for loss calculations
# It works by feedforwarding the input image through the network and generate activations
# Then obtain the average and sum of those outputs
def calc_loss(image, model):
  img_batch = tf.expand_dims(image, axis=0) # Convert into batch format
  layer_activations = model(img_batch) # Run the model
  print('ACTIVATION VALUES (LAYER OUTPUT) =\n', layer_activations)

  losses = [] # accumulator to hold all the losses
  for act in layer_activations:
    loss = tf.math.reduce_mean(act) # calculate mean of each activation
    losses.append(loss)

  print('LOSSES (FROM MULTIPLE ACTIVATION LAYERS) = ', losses)
  print('SUM OF ALL LOSSES (FROM ALL SELECTED LAYERS)= ', tf.reduce_sum(losses))

  return  tf.reduce_sum(losses) # Calculate sum

loss = calc_loss(tf.Variable(Sample_Image), deepdream_model)
print("check calc_loss (should have value): {loss}")

# IMPLEMENT DEEP DREAM ALGORITHM - STEP #2 (CALCULATE THE GRADIENT)
# In this step, we will rely on the loss that has been calculated in the previous step
# and calculate the gradient with respect to the given input image and then add it to the input original image.
# Doing so will result in feeding images that increasingly excite the neurons and generate more dreamy like images!
# When you annotate a function with tf.function, the function can be called like any other python defined function.
# The benefit is that it will be compiled into a graph so it will be much faster and could be executed over TPU/GPU
@tf.function
def deepdream(model, image, step_size):
    with tf.GradientTape() as tape:
      # This needs gradients relative to `img`
      # `GradientTape` only watches `tf.Variable`s by default
      tape.watch(image)
      loss = calc_loss(image, model) # call the function that calculate the loss

    # Calculate the gradient of the loss with respect to the pixels of the input image.
    # The syntax is as follows: dy_dx = g.gradient(y, x)
    gradients = tape.gradient(loss, image)

    print('GRADIENTS =\n', gradients)

    # tf.math.reduce_std computes the standard deviation of elements across dimensions of a tensor
    gradients /= tf.math.reduce_std(gradients)

    # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
    # You can update the image by directly adding the gradients (because they're the same shape!)
    image = image + gradients * step_size
    image = tf.clip_by_value(image, -1, 1)

    return loss, image


def run_deep_dream_simple(model, image, steps = 100, step_size = 0.01):
  # Convert from uint8 to the range expected by the model.
  image = tf.keras.applications.inception_v3.preprocess_input(image)

  for step in range(steps):
    loss, image = deepdream(model, image, step_size)

    if step % 100 == 0:
      plt.figure(figsize=(12,12))
      plt.imshow(deprocess(image))
      plt.show()
      print ("Step {}, loss {}".format(step, loss))

  plt.figure(figsize=(12,12))
  plt.imshow(deprocess(image))
  plt.show()

  return deprocess(image)


# This helper function normalise image.
def deprocess(image):
  image = 255*(image + 1.0)/2.0
  return tf.cast(image, tf.uint8)

# check gradient works
dream_img = run_deep_dream_simple(model = deepdream_model, image = Sample_Image)


# (VIDEO) APPLY DEEPDREAM TO GENERATE A SERIES OF IMAGES
# Define constants
x_size = 910
y_size = 605
max_count = 50

# This helper function loads an image and returns it as a numpy array of floating points
def load_image(filename):
    image = Image.open(filename)
    return np.float32(image)

for i in range(0, max_count):
    # file last image location, to potentially start starting
    if os.path.isfile('data/img_{}.jpg'.format(i+1)):
        print("data/img_{} present already, continue fetching the frames...".format(i+1))

    else:
        # Call the load image funtion
        img_result = load_image('data/img_{}.jpg'.format(i))

        # Zoom the image
        x_zoom = 2 # this indicates how quick the zoom is
        y_zoom = 1

        # Chop off the edges of the image and resize the image back to the original shape. This gives the visual changes of a zoom
        img_result = img_result[0+x_zoom : y_size-y_zoom, 0+y_zoom : x_size-x_zoom]
        img_result = cv2.resize(img_result, (x_size, y_size))

        # Adjust the RGB value of the image
        img_result[:, :, 0] += 2  # red
        img_result[:, :, 1] += 2  # green
        img_result[:, :, 2] += 2  # blue

        # Deep dream model
        img_result = run_deep_dream_simple(model = deepdream_model, image = img_result, steps = 500, step_size = 0.001)

        # Clip the image, convert the datatype of the array, and then convert to an actual image.
        img_result = np.clip(img_result, 0.0, 255.0)
        img_result = img_result.astype(np.uint8)
        result = Image.fromarray(img_result, mode='RGB')

        # Save all the frames in the dream location
        result.save('data/img_{}.jpg'.format(i+1))

        if i > max_count:
            break
