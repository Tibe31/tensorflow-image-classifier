import matplotlib.pyplot as plt


#show all the images in the training batch
def show_augmentations(BATCH_SIZE,train_generator):
  x= train_generator.next()
  for i in range(0,BATCH_SIZE-1):
    image = x[0][i]
    plt.imshow(image)
    plt.show()
