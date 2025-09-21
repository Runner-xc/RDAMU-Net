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

**2. Configure Training Parameters**
```bash
python train.py \
   --data_path your_dataset.csv \
   --data_root_path csv_root_path \
   --model model 
   ```

### 📦 Data Preparation
#### 📂 File Structure
```text
datasets/
├── csv/           # Data paths
├── images/        # Raw images 
└── masks/         # masks
```

#### 🔄 Generate CSV File
```python
from utils.my_data import save_sem_paths_to_csv
save_sem_paths_to_csv("root_path", "csv_path", "csv_name")
```

### ©️ License
This project is licensed under the [Apache License 2.0](./LICENSE).

### 📕 Statement
The code in this repository is for academic research use only. Commercial use is strictly prohibited without permission.
