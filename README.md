### Overview

This project covers the training (from scratch) of a transformer-based language model for the named entity recognition task (NER). 
The idea was to automate the process of annotating data describing machine learning (ML) algorithms, mainly in the form of papers.

We chose three key entity types which we wanted to identify in the papers: 

1. Task (e.g., node clustering)
2. Method (e.g., transformers)
3. Material (e.g., IAM dataset)



### Model architecture and training

The model was built using the Keras library. The model architecture consisted of an Embedding layer, a Transformer block layer, and two pairs
of Fully Connected and Dropout Layers, totaling 133,736 trainable (model) parameters. The Transformer block layer includes a Multihead Attention Layer, followed by a Fully
Connected Layer, Normalization, and Dropout Layers. The Multihead Attention Layer allows the model to jointly attend to information from different representation subspaces.

To train the model we used the sparse categorical cross entropy loss function which
is commonly used for multi-class classification problems because it outputs a probability
distribution over the class labels. The SCIERC corpus was split into 90% training data
(450 paper abstracts) and 10% validation data (50 paper abstracts). We used the Adam
optimization algorithm to train the model over 100 epochs.

### More info.

The development process is documented in more detail in Chapter 7 of my MSc thesis (p. 81), available at: https://drive.google.com/file/d/1vyV6YlN47wOhkFUvjNq_JC63hnleZo9y/view?usp=sharing.
