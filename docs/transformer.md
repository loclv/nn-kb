# Transformer Architectures

A Transformer is a deep learning architecture primarily used in natural language processing (NLP). Unlike previous recurrent neural networks (RNNs) and Long Short-Term Memory (LSTM) models, transformers do not process data sequentially but use a multi-head attention mechanism.

## 1. Definition

A **Transformer** is a deep learning model that adopts the mechanism of **self-attention**, differentially weighting the significance of each part of the input data. It is primarily used in the fields of natural language processing (NLP) and computer vision (CV). 

*   **Parallelization:** Unlike RNNs, Transformers do not require that the sequential data be processed in order. This allows for much more parallelization than RNNs and therefore reduces training times.
*   **Scalability:** This architecture has enabled the training of extremely large models, such as BERT, GPT, and modern Large Language Models (LLMs).

## 2. Mechanism (Attention)

The core innovation of the Transformer is the **Scaled Dot-Product Attention** mechanism.

*   **Query (Q), Key (K), and Value (V):** Each input token is transformed into three vectors. The model calculates how much "attention" to pay to other tokens by comparing the Query of the current token with the Keys of all other tokens.
*   **Multi-Head Attention:** Instead of performing a single attention function, the model performs multiple attention functions (heads) in parallel. This allows the model to jointly attend to information from different representation subspaces at different positions.

## 3. Architecture

The standard Transformer utilizes an **Encoder-Decoder** structure:

*   **Encoder:** Consists of a stack of identical layers. Each layer has two sub-layers: a multi-head self-attention mechanism and a simple, position-wise fully connected feed-forward network. It creates a contextualized representation of the input.
*   **Decoder:** Also consists of a stack of identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.
*   **Positional Encoding:** Since the model contains no recurrence or convolution, it must inject some information about the relative or absolute position of the tokens in the sequence. This is done by adding "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.

---
*Source: Summarized from Wikipedia (Transformer - Machine Learning Model)*
