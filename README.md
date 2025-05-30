# ForeDoc - MedGemma Radiology

This repository contains the ForeDoc - MedGemma Radiology project. Follow the instructions below to set up and run the application.

### How to Run

To get started with ForeDoc, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/alkari/foredoc.git
    ```

2.  **Navigate into the project directory:**
    ```bash
    cd foredoc/
    ```

3.  **Configure environment variables:**
    Copy the sample environment file and then open it to add your necessary API keys or other configurations.
    ```bash
    cp dot_env_sample .env
    # Now, open .env with your preferred editor (e.g., vi, nano, VS Code) and add your keys.
    vi .env # or nano .env, code .env
    ```

4.  **Create and activate a virtual environment:**
    It's recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv foredoc_venv
    source foredoc_venv/bin/activate
    ```

5.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Start the application:**
    ```bash
    ./start.sh &
    ```
    This command runs the `start.sh` script in the background.

7.  **Bring the process to the foreground (optional):**
    If you want to see the output of the background process, you can bring it to the foreground.
    ```bash
    fg
    ```

8.  **Deactivate the virtual environment:**
    Once you're done using the application, you can exit the virtual environment.
    ```bash
    deactivate
    ```

### Cleaning Up (Optional)

If you need to remove the virtual environment and all installed dependencies, you can do so with the following command:

```bash
rm -rf foredoc_venv/
```
*(Note: This command removes the `foredoc_venv` directory, which contains your virtual environment. Make sure you are in the `foredoc/` directory or adjust the path accordingly.)*

---
