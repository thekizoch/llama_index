import os


def load_env(env_path=".env"):
    """
    Load environment variables from a .env file.
    """
    if not os.path.exists(env_path):
        print(f"Warning: {env_path} file not found.")
        return

    with open(env_path) as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()
