# 🤝 Contributing to Smart Waste Classifier

First off, thank you for considering contributing to **Smart Waste Classifier**! Projects like this thrive thanks to community involvement and open collaboration in deep learning, sustainability, and computer vision.

Please take a moment to review this guide to streamline the contribution process.

---

## 🛠️ Development Setup

### 1. Fork & Clone the Repository
```bash
git clone https://github.com/Snigdha-0210/Smart_Waste_Classifier.git
cd Smart_Waste_Classifier
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest pytest-cov
```

---

## 🧪 Running Unit Tests

We use `pytest` for automated unit testing covering model architectures, OOD uncertainty metrics, taxonomy integrity, and pipeline utilities:

```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage summary
pytest tests/ -v --cov=.
```

---

## 🚀 Development Workflow

1. **Branch Naming**: Use descriptive branch names:
   - `feat/feature-name`
   - `fix/bug-description`
   - `docs/documentation-update`
   - `model/dataset-v4-merger`

2. **Code Standards**:
   - Write clean, well-commented Python code following PEP 8 conventions.
   - Avoid committing large raw dataset files, checkpoints, or cached weights directly to Git. Check `.gitignore` before staging files.

3. **Verify App Functionality**:
   - Web App: `streamlit run app.py`
   - ResNet Classification: `python evaluate_resnet.py`
   - YOLOv11 Detection: `python detect_waste.py`

4. **Submit a Pull Request**:
   - Push your branch to GitHub.
   - Open a PR against `main` using the provided [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).
   - Ensure the CI automated test suite passes.

---

## 🌿 Waste Stream Taxonomy

All classification and detection contributions must conform to the 6 standardized municipal waste classes:

| Class Index | Waste Stream | Target Materials |
| :---: | :--- | :--- |
| `0` | **Cardboard** | Corrugated boxes, packaging cartons, egg cartons |
| `1` | **Food Organics** | Fruit/veg peels, food leftovers, coffee grounds, eggshells |
| `2` | **Glass** | Clear/colored bottles, jars, glass containers |
| `3` | **Metal** | Aluminum cans, tin cans, foil, bottle caps |
| `4` | **Paper** | Newspapers, office sheets, flyers, paper bags |
| `5` | **Plastic** | PET bottles, HDPE containers, plastic wrappers, cups |

---

## 📜 Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.
