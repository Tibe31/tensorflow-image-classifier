



def show_augmentations(BATCH_SIZE,x):
  x= train_generator.next()
  for i in range(0,BATCH_SIZE-1):
    image = x[0][i]
    plt.imshow(image)
    plt.show()
