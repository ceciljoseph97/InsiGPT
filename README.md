# InsiGPT with Multi-Head Attention
InsiGPT is a language model based on the GPT architecture, trained using PyTorch. InsiGPT incorporates Multi-Head Attention mechanism, which enables the model to attend to different parts of the input sequence in parallel and capture more complex relationships between the tokens.

## Model Architecture
InsiGPT architecture is based on the Transformer model, which uses self-attention to capture the dependencies between tokens in the input sequence. The Multi-Head Attention mechanism is used in InsiGPT to allow the model to attend to multiple, different representations of the input sequence, which are projected into several smaller subspaces. These subspaces are then concatenated and linearly transformed to produce the final attention output.

The InsiGPT model consists of several layers of self-attention and feedforward neural networks, with residual connections and layer normalization applied at each layer. The final layer outputs a probability distribution over the vocabulary, which is used to generate the next token in the sequence during training and inference.


`in Development`
