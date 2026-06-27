from config_loader import load_config


def test_load_config():
    assert load_config("name: demo")["name"] == "demo"

