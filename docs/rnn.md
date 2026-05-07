# Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are a class of artificial neural networks where connections between nodes can create a cycle, allowing output from some nodes to affect subsequent input to the same nodes. This allows it to exhibit temporal dynamic behavior.

## 1. Definition

Recurrent Neural Networks (RNNs) are designed for processing **sequential data**, such as text, speech, and time series, where the order of elements is critical. 

*   **Mechanism:** Unlike standard feedforward networks, RNNs utilize "recurrent" connections. The output of a neuron at one time step is fed back as an input at the next time step.
*   **Hidden State:** This feedback loop creates a **hidden state**—a form of internal memory that is updated at each step, allowing the network to capture temporal dependencies and patterns within a sequence.

## 2. Architectures (LSTM & GRU)

To address the "vanishing gradient problem" (the difficulty of learning long-term dependencies), more advanced architectures were developed:

*   **Long Short-Term Memory (LSTM):** Invented in 1995, it is the most widely used RNN variant. It uses "gates" (including an input gate, output gate, and **forget gate**) to control the flow of information, allowing gradients to persist over thousands or even millions of time steps without vanishing.
*   **Gated Recurrent Unit (GRU):** Introduced in 2014 as a simplified version of LSTM. It uses fewer parameters (omitting the output gate) and is more computationally efficient, while generally offering similar performance to LSTM in tasks like speech and music modeling.

## 3. Applications

RNNs are utilized across a wide variety of domains, including:

*   **Natural Language & Speech:** Machine translation, speech recognition, speech synthesis, and grammar learning.
*   **Computer Vision:** Handwriting recognition and human action recognition.
*   **Forecasting:** Time series prediction (e.g., energy consumption, inflation), anomaly detection, and rhythm learning.
*   **Robotics & Medicine:** Robot control, brain–computer interfaces, protein homology detection, and medical care pathway prediction.
*   **Science & Engineering:** Music composition and predicting fusion plasma disruptions in nuclear reactors.

---
*Source: Summarized from Wikipedia (Recurrent Neural Network)*
