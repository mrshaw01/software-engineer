# Deep Learning

A subset of machine learning methods that use deep neural networks (DNNs) as models.
A DNN consists of multiple layers.

**Common architectures:**

- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformers (e.g., BERT, T5, GPT)

**Example:** MNIST classification using LeNet.

# Tensor

A **tensor** in deep learning informally refers to a multi-dimensional array.

- Formally, this differs from the mathematical or physical definition of tensors.

# Layer

A **layer** (also called a primitive or operator) can be modeled as a function:

$$Y = L(X_1, X_2, \dots, X_N)$$

- $X_i$: the $i^{th}$ input tensor
- $Y$: the output tensor

Some layers can also produce multiple outputs.

A DNN can be represented as a **data-flow graph**:

- **Nodes** represent layers.
- **Edges** represent data dependencies between layers.

# Activations and Parameters

- **Activations:** Temporary tensors (outputs of layers).
- **Parameters (weights):** Persistent tensors that define the model.

# Inference

DNNs operate in two phases: **inference** and **training**.

**Inference phase:**

1. Input data are fed into the DNN.
2. Data propagate through the data-flow graph (forward pass).
3. Output data are produced.
4. Model parameters remain fixed (already trained).

# Training

**Training phase:**

1. Perform a forward pass to compute the output.
2. Evaluate the output using a **loss function**, which produces a scalar value called the **loss**.

   - The loss decreases as the model output gets closer to the expected value.
   - Examples: Mean Squared Error (MSE), Cross-Entropy Loss.

3. Update the model parameters to minimize the loss, typically using **gradient descent (GD)**.

# Gradient Descent

Gradient descent optimizes the model parameters $\theta$ as follows:

$$\theta \leftarrow \theta - \gamma \cdot \nabla_\theta l(\theta)$$

Where:

- $\theta$: model parameters
- $l(\theta)$: loss function
- $\nabla_\theta l(\theta)$: gradient of the loss with respect to $\theta$
- $\gamma$: learning rate (controls the update size)

**Key idea:**
The gradient indicates the direction of steepest increase in the loss.
By subtracting the gradient, parameters are updated in the direction that **decreases the loss most quickly**.
