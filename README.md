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
