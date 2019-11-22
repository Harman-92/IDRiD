# IDRiD
Repository consisting of computer vision and machine learning techniques on INRiD dataset

### Setup of the project 

Import the IDRiD dataset in the resources tab of the repository.

        -> resources 
              -> Task1
                   -> Test
                   -> Train
              -> Task2
                   -> Test
                   -> Training



The datasets can be obtained from the IDRiD website which is licensed under a 
Creative Commons Attribution 4.0 International License. 


## Implementation Techniques:

1) Task 1:
      1) 
      
      
      
## Dataset:

1) The original dataset given is of the shape (54, 2, 2848, 4288, 3)

    The positions for the shape is (I, S, H, W, C)
    
    `
    where 
          
          I is the Input size of the dataset
          S is the number of datasplits and in our case it is the train and test data
          H is the Height of an image
          W is the Width of an image
          C is the number of channels
     `
      
## Commands:

To run the project program, type the following command:

`python main.py -m Train -dir original_retinal_images -imf jpg`