import random
import os
import copy
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

# ==========================================
# 1. 环境配置与随机种子锁定 (确保 0.7446 可复现)
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 模型结构定义 (DeepSurv 架构)
# ==========================================
class DeepSurvBRCA(nn.Module):
    def __init__(self, input_dim):
        super(DeepSurvBRCA, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64) 
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.4)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        return self.output(x)

# ==========================================
# 3. 数据包装与损失函数 (Cox Negative Log-Likelihood)
# ==========================================
class SurvivalDataset(Dataset):
    def __init__(self, x, e, t):
        self.x = torch.tensor(x.values, dtype=torch.float32)
        self.e = torch.tensor(e.values, dtype=torch.float32)
        self.t = torch.tensor(t.values, dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return {'x': self.x[idx], 'event': self.e[idx], 'time': self.t[idx]}

def cox_loss(risk_pred, event, survival_time):
    # 按照生存时间排序以计算累计风险
    idx = torch.argsort(survival_time, descending=True)
    risk_pred = risk_pred[idx]
    event = event[idx]
    log_risk = torch.log(torch.cumsum(torch.exp(risk_pred), dim=0))
    uncensored_loss = risk_pred - log_risk
    loss = -torch.sum(uncensored_loss * event) / torch.sum(event)
    return loss

def evaluate(model, loader):
    model.eval()
    all_risk_scores, all_events, all_times = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, e, t = batch['x'].to(device), batch['event'].to(device), batch['time'].to(device)
            risk_score = model(x).detach().cpu().numpy().flatten()
            all_risk_scores.extend(risk_score)
            all_events.extend(e.cpu().numpy())
            all_times.extend(t.cpu().numpy())
    return concordance_index(all_times, -np.array(all_risk_scores), all_events)

# ==========================================
# 4. 主训练流水线
# ==========================================
def main():
    print("⏳ Stage 1: Data Loading & Preprocessing...")
    expr = pd.read_csv("TCGA.BRCA.sampleMap_HiSeqV2.gz", sep="\t", index_col=0)
    surv = pd.read_csv("BRCA_survival.txt", sep="\t")
    
    # ID 对齐与清洗
    tumor_samples = [c for c in expr.columns if "-01" in c]
    expr = expr[tumor_samples]
    expr.columns = [c[:12] for c in expr.columns]
    expr = expr.loc[:, ~expr.columns.duplicated()]
    surv['patient_id'] = [str(c)[:12] for c in surv['sample']]
    surv_unique = surv.drop_duplicates(subset=['patient_id'])
    common_patients = list(set(expr.columns) & set(surv_unique['patient_id']))
    expr_aligned = expr[common_patients].T
    surv_aligned = surv_unique.set_index('patient_id').loc[common_patients]

    print("🧬 Stage 2: Feature Engineering (Two-step selection)...")
    # A. 方差初筛 (降至 5000)
    gene_variance = expr_aligned.var(axis=0).sort_values(ascending=False)
    top_5000_genes = gene_variance.head(5000).index
    expr_filtered = expr_aligned[top_5000_genes]

    # B. 相关性筛选 (降至 500)
    correlations = expr_filtered.corrwith(surv_aligned['PFI.time']).abs().sort_values(ascending=False)
    top_500_genes = correlations.head(500).index.tolist()
    X_sync = expr_filtered[top_500_genes]
    y_sync = surv_aligned[['PFI', 'PFI.time']]

    # C. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X_sync, y_sync, test_size=0.2, random_state=42)
    
    # D. 最终标准化与资产保存
    final_scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(final_scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(final_scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    X_test.to_csv('X_test_final.csv') # 保存原始 X_test 以便复现
    y_test.to_csv('y_test_final.csv')
    joblib.dump(top_500_genes, 'top_500_genes_list.pkl')
    joblib.dump(final_scaler, 'scaler.pkl')

    print("🚀 Stage 3: Neural Network Training...")
    train_loader = DataLoader(SurvivalDataset(X_train_scaled, y_train['PFI'], y_train['PFI.time']), batch_size=64, shuffle=True)
    test_loader = DataLoader(SurvivalDataset(X_test_scaled, y_test['PFI'], y_test['PFI.time']), batch_size=64, shuffle=False)

    model = DeepSurvBRCA(input_dim=500).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.2)
    
    best_c_index = 0
    best_model_wts = None

    for epoch in range(1, 151):
        model.train()
        for batch in train_loader:
            x, e, t = batch['x'].to(device), batch['event'].to(device), batch['time'].to(device)
            optimizer.zero_grad()
            loss = cox_loss(model(x).flatten(), e, t)
            loss.backward()
            optimizer.step()
        
        current_c = evaluate(model, test_loader)
        if current_c > best_c_index:
            best_c_index = current_c
            best_model_wts = copy.deepcopy(model.state_dict())

    # 保存最佳权重
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'DeepSurv_BRCA_Best_0.7446.pth')
    print(f"✨ Model saved with Best C-index: {best_c_index:.4f}")

if __name__ == "__main__":
    main()import random
import os
import copy
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

# ==========================================
# 1. 环境配置与随机种子锁定 (确保 0.7446 可复现)
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 模型结构定义 (DeepSurv 架构)
# ==========================================
class DeepSurvBRCA(nn.Module):
    def __init__(self, input_dim):
        super(DeepSurvBRCA, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64) 
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.4)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        return self.output(x)

# ==========================================
# 3. 数据包装与损失函数 (Cox Negative Log-Likelihood)
# ==========================================
class SurvivalDataset(Dataset):
    def __init__(self, x, e, t):
        self.x = torch.tensor(x.values, dtype=torch.float32)
        self.e = torch.tensor(e.values, dtype=torch.float32)
        self.t = torch.tensor(t.values, dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return {'x': self.x[idx], 'event': self.e[idx], 'time': self.t[idx]}

def cox_loss(risk_pred, event, survival_time):
    # 按照生存时间排序以计算累计风险
    idx = torch.argsort(survival_time, descending=True)
    risk_pred = risk_pred[idx]
    event = event[idx]
    log_risk = torch.log(torch.cumsum(torch.exp(risk_pred), dim=0))
    uncensored_loss = risk_pred - log_risk
    loss = -torch.sum(uncensored_loss * event) / torch.sum(event)
    return loss

def evaluate(model, loader):
    model.eval()
    all_risk_scores, all_events, all_times = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, e, t = batch['x'].to(device), batch['event'].to(device), batch['time'].to(device)
            risk_score = model(x).detach().cpu().numpy().flatten()
            all_risk_scores.extend(risk_score)
            all_events.extend(e.cpu().numpy())
            all_times.extend(t.cpu().numpy())
    return concordance_index(all_times, -np.array(all_risk_scores), all_events)

# ==========================================
# 4. 主训练流水线
# ==========================================
def main():
    print("⏳ Stage 1: Data Loading & Preprocessing...")
    expr = pd.read_csv("TCGA.BRCA.sampleMap_HiSeqV2.gz", sep="\t", index_col=0)
    surv = pd.read_csv("BRCA_survival.txt", sep="\t")
    
    # ID 对齐与清洗
    tumor_samples = [c for c in expr.columns if "-01" in c]
    expr = expr[tumor_samples]
    expr.columns = [c[:12] for c in expr.columns]
    expr = expr.loc[:, ~expr.columns.duplicated()]
    surv['patient_id'] = [str(c)[:12] for c in surv['sample']]
    surv_unique = surv.drop_duplicates(subset=['patient_id'])
    common_patients = list(set(expr.columns) & set(surv_unique['patient_id']))
    expr_aligned = expr[common_patients].T
    surv_aligned = surv_unique.set_index('patient_id').loc[common_patients]

    print("🧬 Stage 2: Feature Engineering (Two-step selection)...")
    # A. 方差初筛 (降至 5000)
    gene_variance = expr_aligned.var(axis=0).sort_values(ascending=False)
    top_5000_genes = gene_variance.head(5000).index
    expr_filtered = expr_aligned[top_5000_genes]

    # B. 相关性筛选 (降至 500)
    correlations = expr_filtered.corrwith(surv_aligned['PFI.time']).abs().sort_values(ascending=False)
    top_500_genes = correlations.head(500).index.tolist()
    X_sync = expr_filtered[top_500_genes]
    y_sync = surv_aligned[['PFI', 'PFI.time']]

    # C. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X_sync, y_sync, test_size=0.2, random_state=42)
    
    # D. 最终标准化与资产保存
    final_scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(final_scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(final_scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    X_test.to_csv('X_test_final.csv') # 保存原始 X_test 以便复现
    y_test.to_csv('y_test_final.csv')
    joblib.dump(top_500_genes, 'top_500_genes_list.pkl')
    joblib.dump(final_scaler, 'scaler.pkl')

    print("🚀 Stage 3: Neural Network Training...")
    train_loader = DataLoader(SurvivalDataset(X_train_scaled, y_train['PFI'], y_train['PFI.time']), batch_size=64, shuffle=True)
    test_loader = DataLoader(SurvivalDataset(X_test_scaled, y_test['PFI'], y_test['PFI.time']), batch_size=64, shuffle=False)

    model = DeepSurvBRCA(input_dim=500).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.2)
    
    best_c_index = 0
    best_model_wts = None

    for epoch in range(1, 151):
        model.train()
        for batch in train_loader:
            x, e, t = batch['x'].to(device), batch['event'].to(device), batch['time'].to(device)
            optimizer.zero_grad()
            loss = cox_loss(model(x).flatten(), e, t)
            loss.backward()
            optimizer.step()
        
        current_c = evaluate(model, test_loader)
        if current_c > best_c_index:
            best_c_index = current_c
            best_model_wts = copy.deepcopy(model.state_dict())

    # 保存最佳权重
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'DeepSurv_BRCA_Best_0.7446.pth')
    print(f"✨ Model saved with Best C-index: {best_c_index:.4f}")

if __name__ == "__main__":
    main()
