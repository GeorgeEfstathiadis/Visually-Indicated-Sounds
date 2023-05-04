# Visually-Indicated-Sounds

CV Project

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
    * [ ] Model training
      * [x] Training loop
      * [ ] TODO: fix get_random that sometimes generates empty audio for non-matching video/audio tracks.
  * [ ] Bonus Model: apply convolution to the time dimension (frames) of the video by adding initial layer in the pretrained ResNet
    * [ ] not enough memory to work with c\*h\*w channels, need to convolute spatially first
* [ ] create new data outside the drumstick dataset to see how it generalizes

## Useful links:

* https://github.com/emilyzfliu/vis-sounds/tree/main
