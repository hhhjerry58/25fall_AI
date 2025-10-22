# Installation

---

## 1.Install Miniconda

---

## 2. create and activate conda enviornment

```powershell
conda create -n 25fall_ai python=3.10 -y
conda activate 25fall_ai
```

---

## 3. Install dependencies:

After activation, run following command in terminal

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install matplotlib
```
---

# File organization

The files are organized as follows:

```powershell
\data # MNIST dataset
\source1.py # Answer for Q5.2
\source2.py # Answer for Q5.3
\source3.py # Answer for Q5.5
```

---
# Usage
```powershell
cd codes
python source{i}.py # Running command for each python file
```
In source2.py we could modify b to change loss function.




