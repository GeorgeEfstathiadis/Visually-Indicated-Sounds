# Visually-Indicated-Sounds

CV Project

Current best test accuracy: 0.45833 (fusion_2)

## What needs to be done:

* [x] create a dataloader for the dataset
  * [ ] could use speedup for video/audio loading
  * [x] check video/audio is in sync
* [x] create training-validation-test sets
* [ ] discriminator network to see if video/audio is real or fake
  * [ ] Model1: images -> ResNet -> GRU (for the video) combined with audio (stereo handle both channels) -> ResNet 
    * concat output (512 + 512) and parse through a linear layer
  * [ ] Model2: Siamese fusion network, same as model1 but with same weights (using resnet18)
    * [x] Model structure
    * [x] Model training
      * [x] Training loop
      * [x] First run successful! (see logs)
      * [ ] Update structure & parameters until satisfying results
  * [ ] Bonus Model: apply convolution to the time dimension (frames) of the video by adding initial layer in the pretrained ResNet
    * [ ] not enough memory to work with c\*h\*w channels, need to convolute spatially first
* [ ] create new data outside the drumstick dataset to see how it generalizes

### Report

* [ ] Update Abstract (after finishing the report on limitations and results)
* [x] Introduction (can combine with related work if we need the pages)
* [x] Related Work
* [ ] Methods
  * [x] Dataset
  * [ ] Model Architecture - Change to write about the Siamese network, non-explicit pre-trained models (remove resnet mention, keep it more generic -> will move this to results since we tried more different architectures), focus on fusion network strategy
* [ ] Results
  * [ ] Data Loading (write about video/audio loading speedup techniques, removing information for faster data loading in training etc.)
  * [ ] Overfitting (write on efforts to avoid overfitting and get good convergent results)
    * [ ] Data Augment (write about data augmentations used, different things tried and results)
    * [ ] Model Augment (write about model augmentations used, different things tried and results)
* [ ] Discussion
  * [x] Aims (write about aims of the project, what we wanted to achieve)
  * [ ] Limitations (write about limitations of the dataset, of the model, of the training process)
  * [ ] Future Work (write about future work, what could be done to improve the results, what could be done to improve the dataset, what could be done to improve the model, What steps would you have taken had you continued working on it?)
* [ ] Contributions (write about contributions of each member)
* [ ] Fix graph of network architecture (Fusion to be more generic components)
* [ ] Training loss; accuracies graphs? (maybe not since not that interesting; just shown overfitting; maybe appendix??)

## Useful links:

* https://github.com/emilyzfliu/vis-sounds/tree/main
