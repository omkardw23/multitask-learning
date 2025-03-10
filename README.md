Multi-Task Learning with Sentence Transformers
Project Overview
This project explores the implementation of a multi-task learning model using a pre-trained transformer for sentence classification and named entity recognition (NER). By leveraging BERT-based embeddings, the model effectively encodes sentences into meaningful representations and applies separate task-specific heads for classification and entity recognition.

The goal is to build a modular, scalable, and GPU-accelerated training pipeline that can be easily deployed and reproduced using Docker. The repository contains all necessary scripts to train, evaluate, and infer from the model, along with a fully containerized setup to ensure a seamless execution environment.

Model Implementation
The sentence transformer model is built on BERT (bert-base-uncased), which is used to generate contextualized sentence embeddings. These embeddings are then used in a multi-task setting, where the model performs sentence classification and named entity recognition (NER) simultaneously.

For sentence classification, the [CLS] token's representation is extracted from the transformer’s output and passed through a classification head to predict sentence types. For NER, the token-wise embeddings are used to classify each token into predefined entity categories. This architecture allows the model to share learned representations while maintaining task-specific adaptability.

To efficiently handle the two tasks, the model is structured with a shared transformer backbone and two independent task-specific heads, ensuring that the tasks benefit from shared language understanding while optimizing for their respective objectives.

Multi-Task Learning Expansion
Multi-task learning (MTL) introduces the challenge of balancing multiple objectives within a shared learning framework. To support MTL, the architecture extends the sentence transformer with separate output heads, one for sentence classification and another for named entity recognition. The classification head takes the [CLS] token embedding, while the NER head processes token-level embeddings to predict named entities.

Each task has its own loss function, and during training, the combined loss is used to update the shared backbone and the respective task heads. This approach enables parameter sharing while allowing task-specific learning, ultimately leading to improved generalization.

Training Considerations
Several training strategies were evaluated to determine the most efficient approach. One possibility is freezing the entire network, which allows only the classification and NER heads to be trained. This approach is suitable when working with small datasets where retraining the backbone would lead to overfitting. However, it limits the model’s ability to learn domain-specific features.

Another approach is freezing only the transformer backbone while training the classification and NER heads. This method retains the pre-trained knowledge of the transformer while adapting it for the given tasks. It is particularly useful when the dataset is small but has significant differences from the original pre-training corpus.

A more flexible strategy is to freeze only one task head, allowing the model to leverage pre-trained knowledge for one task while fine-tuning the other. This approach is beneficial when one of the tasks has a significantly larger dataset, preventing catastrophic forgetting while improving task-specific performance.

For transfer learning, a gradual unfreezing approach is adopted, where the last few layers of the transformer backbone are unfrozen first, allowing the model to learn new task-specific features without completely forgetting its pre-trained knowledge. The pre-trained model chosen for this project is bert-base-uncased, as it provides a strong general-purpose embedding model with broad applicability to NLP tasks.

Training Implementation
The training loop is designed to efficiently handle multi-task learning, alternating between batches for sentence classification and NER. A combined loss function is computed, ensuring that updates are made to both tasks simultaneously. Loss balancing is handled dynamically to prevent one task from dominating the learning process.

During each epoch, batches from the sentence classification dataset and the NER dataset are passed through the transformer. The [CLS] token representation is extracted for classification, while per-token embeddings are used for entity recognition. The cross-entropy loss is calculated separately for both tasks, and their gradients are accumulated before updating the model parameters.

For evaluation, accuracy metrics are used for sentence classification, while F1-score and token-level accuracy are used for NER. The training loop includes detailed logging and progress tracking using tqdm, ensuring visibility into the model’s learning process.

Dockerized Deployment and Automation
To ensure reproducibility and ease of deployment, the project is fully Dockerized, allowing seamless execution across different environments. The Dockerfile defines a PyTorch-based GPU environment with all required dependencies. The model can be trained and deployed inside a containerized setup, ensuring consistency in results.

The training and inference processes can be automated using a shell script that builds the Docker image, runs training on GPU, and performs inference using the trained model. Additionally, Docker Compose can be used to orchestrate multiple services, allowing training and inference to run separately.

How to Run the Project
The project requires Python 3.9+, PyTorch, and Hugging Face Transformers. All dependencies are listed in requirements.txt, making it easy to set up the environment.

To train the model locally, simply run:

For training:
```python scripts/train.py```

For inference:
```python scripts/inference.py```

To run everything inside Docker, first build the image:
```docker build -t multitask-transformer .```

Once built, training can be started using:
```docker run --gpus all --rm multitask-transformer python scripts/train.py```

For inference:
```docker run --gpus all --rm multitask-transformer python scripts/inference.py```

If debugging inside the container is required, an interactive session can be launched with:
```docker run -it --gpus all --rm multitask-transformer /bin/bash```