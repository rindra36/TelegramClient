# Binary Options Trading Bot

A Python-based automated trading bot that integrates with Telegram for binary options trading signals and execution.

## 📋 Prerequisites

- Python 3.7+
- pip (Python package installer)
- Git
- Rust (for building BinaryOptionsTools-V2)

## 🚀 Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

2. Install project dependencies:
```bash
pip install -r requirements.txt
```

3. Set up required libraries:
```bash
mkdir lib && cd lib

# Install BinaryOptionsTools-V2
git clone https://github.com/ChipaDevTeam/BinaryOptionsTools-v2.git
cd BinaryOptionsTools-v2/BinaryOptionsToolsV2
maturin develop -r
cd ..

# Install BinaryOptionsTools-V1
git clone https://github.com/theshadow76/BinaryOptionsTools.git
cd BinaryOptionsTools
pip install .
```

## ⚙️ Configuration

1. Navigate to the `assets` directory
2. Create an `env` directory
3. Copy all the JSON files from `example`
4. Configure the following JSON files:
   - `assets.json`: Trading assets configuration
   - `binomoCredentials.json`: Trading platform credentials
   - `telegramCredentials.json`: Telegram bot credentials
   - `uaCredentials.json`: User authentication details

## 🏃‍♂️ Getting Started

Check out the examples in the `example` directory:
- `getting-started.py`: Basic setup and usage
- `bidding.py`: Trading implementation example
- `main.py`: Full bot implementation

## 📁 Project Structure

```
├── assets/           # Configuration files
├── example/          # Example implementations
├── lib/             # External libraries
├── logs/            # Trading logs
└── src/             # Source code
```

## 📚 Documentation

For detailed usage instructions and API documentation, see the example files in the `example` directory:
- `BinaryOptionsToolsV1/`: Version 1 implementation examples
- `BinaryOptionsToolsV2/`: Version 2 implementation examples with async/sync options

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

Trading binary options involves significant risk. This software is for educational purposes only. Always trade responsibly and at your own risk.