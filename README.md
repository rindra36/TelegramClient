1 - Create a virtual environment : python -m venv .venv
2 - Connect to the virtal environment : source .venv/bin/activate
3 - Install the requirements : pip install -r requirements.txt
4 - Create a lib folder on root project
5 - Go to the lib folder
6 - Clone the project BinaryOptionsTools-V2 : git clone https://github.com/ChipaDevTeam/BinaryOptionsTools-v2.git
7 - Clone the project BinaryOptionsTools-V1 : git clone https://github.com/theshadow76/BinaryOptionsTools.git
8 - Enter folder of BinaryOptionsTools-V2/BinaryOptionsToolsV2 : cd BinaryOptionsTools-v2/BinaryOptionsToolsV2
9 - Build with maturin : maturin develop -r
    a - If needed, upgrade the cargo version : rustup toolchain uninstall stable && rustup toolchain install stable
10 - Enter folder of BinaryOptionsTools-V1 : cd BinaryOptionsTools
11 - Install with pip : pip install .
12 - Do not forget to edit the session name in src/utils.py
13 - To launch in terminal only, do not forget to add to PYTHONPATH the path in .vscode/settings.json