# stocks
# Stocks Project

This project is designed to help you manage and analyze stock data.

## Setup

To set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/oheneralov/stocks.git
    cd stocks
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:

    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source venv/bin/activate
        ```

4. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Running a program
    ```sh
    python main.py
    ```

## Running Tests

To run the tests using pytest, follow these steps:

1. Ensure you are in the virtual environment (see setup steps above).

2. Run the tests:
    ```sh
    pytest
    ```
    or
    pytest test_recurrent.py

This will discover and run all the tests in the project.

## Additional Information

- For more information on pytest, visit the [official documentation](https://docs.pytest.org/en/stable/).
- Make sure to keep your dependencies up to date by running:
    ```sh
    pip install --upgrade -r requirements.txt
    ```

## Few words about recurrent networks
### Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are a class of neural networks designed for processing sequential data, such as time series, text, or speech. Unlike traditional feedforward networks, RNNs have connections that form cycles, allowing information to persist and be passed from one step of the sequence to the next.

#### Types of RNNs

- **Vanilla RNN**: The basic form of RNN, which can suffer from vanishing or exploding gradients when modeling long sequences.
- **Long Short-Term Memory (LSTM)**: An advanced RNN variant that uses gates to better capture long-term dependencies and mitigate vanishing gradients.
- **Gated Recurrent Unit (GRU)**: A simplified version of LSTM with fewer gates, often providing similar performance with less computational cost.
- **Bidirectional RNN**: Processes sequences in both forward and backward directions, improving context understanding for certain tasks.

RNNs are widely used for tasks like language modeling, sequence prediction, and time-series forecasting.
In Keras (and most machine learning frameworks), input features for time-series data are combined 
 into a single 3D array with the shape:

(samples, timesteps, features)
Samples: Number of sequences (how many training examples you have).
Timesteps: Number of time steps per sequence.
Features: Number of features at each time step.
### What is a "Timestep"?
A timestep represents the number of consecutive data points (from the past) that the model looks at to predict the next point.
The reason we need the latest_sequence for prediction in time-series models is that recurrent networks like LSTM use past data (historical context) to make future predictions.

### Why is Historical Data Needed?
The network has learned patterns from past timesteps during training, so it expects a sequence as input to make predictions. If we only provide a single point (like "the current price"), the network wouldn't have enough context to generate accurate predictions.

Analogy
Think of predicting the weather. You wouldn't just look at today's temperature to forecast tomorrow—you'd want to know the past several days' conditions, including trends in temperature, humidity, and wind.

Technical Explanation
When you define the LSTM with input_shape=(timesteps, features), you're explicitly telling it:

"I will always give you data spanning timesteps consecutive points."

During prediction:

The input sequence (latest_sequence) contains the most recent historical data for the timesteps the model needs.
The model uses this sequence to infer patterns and predict the next step.
Simplified Workflow
Training: You train the model on sequences of length timesteps. Example: [100, 101, 102] → predict 103

Prediction: When predicting the next price, provide [102, 103, 104] so the model can predict 105.

Why Not Just One Point?
If you pass just one value (like 104):

The LSTM won't recognize patterns without the required context.
It will produce an error unless reshaped incorrectly (batch_size=1, timesteps=1), leading to poor performance.
What If No Historical Data Exists?
If historical data is unavailable:

Consider a simpler non-recurrent model, like a Dense/Feedforward network, which doesn't require sequences.
Initialize with a constant sequence or a statistical guess for the missing timesteps.

## Deploying to AWS ECS with Fargate
cd terraform/
terraform init
terraform plan -var-file="terraform.tfvars"
terraform apply -var-file="terraform.tfvars"

## Build and push Docker image to ECR
aws ecr get-login-password --region us-east-1 | \
docker login --username AWS --password-stdin <ECR_REPO_URL>

docker build -t mlops-rnn .
docker tag mlops-rnn:latest <ECR_REPO_URL>:latest
docker push <ECR_REPO_URL>:latest

## Do NOT commit terraform.tfvars to Git if it includes sensitive data like secrets or credentials.

## use Airflow (MWAA) as the orchestrator for tasks like:

-Scheduling RNN training (daily/weekly)

-Deploying model artifacts

-Running batch predictions

-Managing data ingestion and preprocessing

### Terraform Setup for MWAA
-Create an S3 bucket for DAGs and logs
-Create an IAM role
-Deploy MWAA environment
-Upload your Airflow DAGs

#### Upload DAGs and Dependencies
Once the environment is created, upload your DAGs and Python packages:
aws s3 cp dags/my_rnn_pipeline.py s3://<your-dag-bucket>/dags/
aws s3 cp requirements.txt s3://<your-dag-bucket>/requirements.tx
DAG (Directed Acyclic Graph) is a collection of tasks with dependencies and a defined execution order.
It tells Airflow:
What to do (e.g., preprocess → train → evaluate → deploy)
When to do it (e.g., every day at midnight)
In what order to do it (based on task dependencies)


## TO  DO
- Add airflow
- ArgoCD
- terraform
- Spark for ELT