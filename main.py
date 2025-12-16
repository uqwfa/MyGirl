import yaml
from pathlib import Path


def load_config():
    config_path = Path("config.yaml")

    if not config_path.exists():
        print("config.yaml not found!")
        exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    from modules.database.database import init_database
    init_database(config)


if __name__ == "__main__":
    main()
