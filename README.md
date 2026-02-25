## RDAMU-Net
---

### 📖 Table of Contents
- [⚡ Quick Start](#-quick-start)
- [📦 Data Preparation](#-data-preparation)
- [©️ License](#️-license)
- [📕 Statement](#-statement)

---

### ⚡ Quick Start
**1. Clone Repository**
```bash
git clone https://github.com/Runner-xc/RDAMU-Net.git
cd RDAMU-Net/
pip install -r requirements.txt
```

***2. Train**
```bash
python train.py \
   --model model_name \
   --train_csv ./train.csv \
   --val_csv   ./val.csv \
   --test_csv  ./test.csv
```


### 📦 Data Preparation
#### 📂 File Structure
```text
datasets/
├── csv/           # Data paths
├── images/        # Raw images 
└── masks/         # masks
```

### ©️ License
This project is licensed under the [Apache License 2.0](./LICENSE).

### 📕 Statement
The code in this repository is for academic research use only. Commercial use is strictly prohibited without permission.
