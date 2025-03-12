python -m venv test_env
source test_env/bin/activate
pip install dist/grampa-0.0.1-py3-none-any.whl
python -c "import grampa; print(grampa.__version__)"


