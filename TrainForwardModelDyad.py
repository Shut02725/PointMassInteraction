import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

from ForwardModelDyad import ForwardDynamicsLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataNormalizer:
    """データの正規化・標準化を管理するクラス"""
    def __init__(self):
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        self.fitted = False
    
    def fit(self, dataset):
        """データセットから統計量を計算（時系列データ対応）"""
        print("データ統計量を計算中...")
        
        inputs = torch.stack([data['input'] for data in dataset])
        outputs = torch.stack([data['output'] for data in dataset])
        
        inputs_flat = inputs.view(-1, inputs.size(-1))
        outputs_flat = outputs.view(-1, outputs.size(-1))
        
        self.input_mean = inputs_flat.mean(dim=0)
        self.input_std = inputs_flat.std(dim=0) + 1e-8
        self.output_mean = outputs_flat.mean(dim=0)
        self.output_std = outputs_flat.std(dim=0) + 1e-8
        
        self.fitted = True
        
        print("\n=== データ統計情報 ===")
        print(f"入力データ: 平均 {self.input_mean.mean():.4f}, 標準偏差 {self.input_std.mean():.4f}")
        print(f"出力データ: 平均 {self.output_mean.mean():.4f}, 標準偏差 {self.output_std.mean():.4f}")
        
        return self
    
    def normalize_input(self, input_data):
        if not self.fitted:
            raise ValueError("Normalizer has not been fitted yet")
        return (input_data - self.input_mean) / self.input_std
    
    def normalize_output(self, output_data):
        if not self.fitted:
            raise ValueError("Normalizer has not been fitted yet")
        return (output_data - self.output_mean) / self.output_std
    
    def denormalize_output(self, normalized_output):
        if not self.fitted:
            raise ValueError("Normalizer has not been fitted yet")
        return normalized_output * self.output_std + self.output_mean
    
    def save(self, filepath):
        normalizer_data = {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'output_mean': self.output_mean,
            'output_std': self.output_std,
            'fitted': self.fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(normalizer_data, f)
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            normalizer_data = pickle.load(f)
        self.input_mean = normalizer_data['input_mean']
        self.input_std = normalizer_data['input_std']
        self.output_mean = normalizer_data['output_mean']
        self.output_std = normalizer_data['output_std']
        self.fitted = normalizer_data['fitted']
        return self

def normalize_dataset(dataset, normalizer):
    """データセットを正規化"""
    normalized_dataset = []
    for data in dataset:
        normalized_input = normalizer.normalize_input(data['input'])
        normalized_output = normalizer.normalize_output(data['output'])
        normalized_dataset.append({
            'input': normalized_input,
            'output': normalized_output
        })
    return normalized_dataset

def train_agent_model(model, dataset_training, dataset_validation, normalizer, 
                     epochs=200, batch_size=32, lr=1e-3, agent_name="Agent", 
                     prediction_mode="last_step", seed=42, verbose=True):
    """
    単一環境用の学習関数
    """
    # 正規化されたデータセットを作成
    train_dataset = normalize_dataset(dataset_training, normalizer)
    val_dataset = normalize_dataset(dataset_validation, normalizer)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=15, verbose=False
    )
    criterion = nn.MSELoss()
    
    model.train()
    
    if verbose:
        print(f"{agent_name} (seed={seed}) 学習開始:")
        print(f"  訓練データ: {len(train_dataset)} samples")
        print(f"  検証データ: {len(val_dataset)} samples")
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        num_batches = 0
        
        train_indices = torch.randperm(len(train_dataset))
        
        for i in range(0, len(train_dataset), batch_size):
            batch_indices = train_indices[i:i+batch_size]
            batch_data = [train_dataset[idx] for idx in batch_indices]
            batch_inputs = torch.stack([data['input'] for data in batch_data]).to(device)
            batch_outputs = torch.stack([data['output'] for data in batch_data]).to(device)

            optimizer.zero_grad()
            
            model_output = model(batch_inputs)
            
            if isinstance(model_output, tuple):
                predictions = model_output[0]
            else:
                predictions = model_output
            
            if prediction_mode == "last_step":
                if batch_outputs.dim() == 3:
                    target_outputs = batch_outputs[:, -1, :]
                else:
                    target_outputs = batch_outputs
                    
                if predictions.dim() == 3:
                    predictions = predictions[:, -1, :]
            else:
                target_outputs = batch_outputs
                if predictions.dim() == 2 and batch_outputs.dim() == 3:
                    seq_len = batch_outputs.size(1)
                    predictions = predictions.unsqueeze(1).expand(-1, seq_len, -1)
                
            loss = criterion(predictions, target_outputs)
            
            mae = torch.mean(torch.abs(predictions - target_outputs))
            output_magnitude = torch.mean(torch.abs(target_outputs))
            if output_magnitude > 1e-8:
                relative_error = mae / output_magnitude
                accuracy = torch.clamp(1 - relative_error, 0, 1)
            else:
                accuracy = torch.tensor(0.0, device=device)
            
            l1_lambda = 1e-5
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            total_loss = loss + l1_lambda * l1_norm
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_accuracy += accuracy.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        avg_train_accuracy = epoch_train_accuracy / num_batches
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        
        # 検証フェーズ
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_accuracy = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_dataset), batch_size):
                batch_data = val_dataset[i:i+batch_size]
                batch_inputs = torch.stack([data['input'] for data in batch_data]).to(device)
                batch_outputs = torch.stack([data['output'] for data in batch_data]).to(device)
                
                model_output = model(batch_inputs)
                
                if isinstance(model_output, tuple):
                    predictions = model_output[0]
                else:
                    predictions = model_output
                
                if prediction_mode == "last_step":
                    if batch_outputs.dim() == 3:
                        target_outputs = batch_outputs[:, -1, :]
                    else:
                        target_outputs = batch_outputs
                        
                    if predictions.dim() == 3:
                        predictions = predictions[:, -1, :]
                else:
                    target_outputs = batch_outputs
                    if predictions.dim() == 2 and batch_outputs.dim() == 3:
                        seq_len = batch_outputs.size(1)
                        predictions = predictions.unsqueeze(1).expand(-1, seq_len, -1)
                    
                loss = criterion(predictions, target_outputs)
                
                mae = torch.mean(torch.abs(predictions - target_outputs))
                output_magnitude = torch.mean(torch.abs(target_outputs))
                if output_magnitude > 1e-8:
                    relative_error = mae / output_magnitude
                    accuracy = torch.clamp(1 - relative_error, 0, 1)
                else:
                    accuracy = torch.tensor(0.0, device=device)
                
                epoch_val_loss += loss.item()
                epoch_val_accuracy += accuracy.item()
                num_val_batches += 1
        
        avg_val_loss = epoch_val_loss / num_val_batches
        avg_val_accuracy = epoch_val_accuracy / num_val_batches
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)
        
        scheduler.step(avg_val_loss)
        
        if verbose and (epoch % 40 == 0 or epoch == epochs - 1):
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{epochs}: Train Loss={avg_train_loss:.6f}, "
                  f"Val Loss={avg_val_loss:.6f}, Train Acc={avg_train_accuracy:.4f}, "
                  f"Val Acc={avg_val_accuracy:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies

def load_dataset_from_directory(save_dir, agent_name):
    """指定されたディレクトリからデータセットを読み込む"""
    if not os.path.exists(save_dir):
        print(f"エラー: ディレクトリ '{save_dir}' が存在しません。")
        return None, None
    
    pkl_files = glob.glob(os.path.join(save_dir, "*.pkl"))
    
    if not pkl_files:
        print(f"エラー: '{save_dir}' にpklファイルが見つかりません。")
        return None, None
    
    print(f"\n=== {agent_name} 用データセット選択 ({save_dir}) ===")
    for i, file_path in enumerate(pkl_files):
        file_name = os.path.basename(file_path)
        print(f"{i+1}: {file_name}")
    
    while True:
        try:
            choice = input(f"\n{agent_name} の学習に使用するファイルを選択してください (1-{len(pkl_files)}): ")
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(pkl_files):
                selected_file = pkl_files[choice_idx]
                break
            else:
                print(f"1から{len(pkl_files)}の間で選択してください。")
        except ValueError:
            print("数字を入力してください。")
    
    print(f"選択されたファイル: {os.path.basename(selected_file)}")
    
    try:
        with open(selected_file, 'rb') as f:
            dataset = pickle.load(f)
        
        if isinstance(dataset, dict):
            if 'training' in dataset and 'validation' in dataset:
                dataset_training = dataset['training']
                dataset_validation = dataset['validation']
            elif 'train' in dataset and 'val' in dataset:
                dataset_training = dataset['train']
                dataset_validation = dataset['val']
            else:
                keys = list(dataset.keys())
                print(f"\n訓練データと検証データのキーを指定してください:")
                for i, key in enumerate(keys):
                    print(f"{i+1}: {key}")
                
                train_choice = int(input("訓練データのキー番号: ")) - 1
                val_choice = int(input("検証データのキー番号: ")) - 1
                
                dataset_training = dataset[keys[train_choice]]
                dataset_validation = dataset[keys[val_choice]]
        
        elif isinstance(dataset, list):
            split_ratio = 0.8
            split_idx = int(len(dataset) * split_ratio)
            dataset_training = dataset[:split_idx]
            dataset_validation = dataset[split_idx:]
        
        else:
            print(f"エラー: 未対応のデータセット形式: {type(dataset)}")
            return None, None
        
        if len(dataset_training) > 0:
            sample = dataset_training[0]
            print(f"データ形式: input {sample['input'].shape}, output {sample['output'].shape}")
        
        print(f"訓練データ: {len(dataset_training)} samples, 検証データ: {len(dataset_validation)} samples")
        
        return dataset_training, dataset_validation
        
    except Exception as e:
        print(f"エラー: データセットの読み込みに失敗しました: {e}")
        return None, None

def plot_multi_agent_multi_seed_results(all_results, save_dir, epochs):
    """
    複数エージェント×複数シードの学習結果を平均±標準偏差でプロット
    
    Args:
        all_results: {agent_name: {'seeds': [seed1, seed2, ...], 
                                    'train_losses': [[...], [...], ...],
                                    'val_losses': [[...], [...], ...],
                                    'train_acc': [[...], [...], ...],
                                    'val_acc': [[...], [...], ...]}}
        save_dir: 保存先ディレクトリ
        epochs: エポック数
    """
    plt.style.use('default')
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    epoch_list = np.arange(1, epochs + 1)
    
    # 損失のプロット
    for idx, (agent_name, results) in enumerate(all_results.items()):
        color = colors[idx % len(colors)]
        
        # 訓練損失の平均と標準偏差
        train_losses_array = np.array(results['train_losses'])
        train_mean = np.mean(train_losses_array, axis=0)
        train_std = np.std(train_losses_array, axis=0)
        
        # 検証損失の平均と標準偏差
        val_losses_array = np.array(results['val_losses'])
        val_mean = np.mean(val_losses_array, axis=0)
        val_std = np.std(val_losses_array, axis=0)
        
        # 訓練損失のプロット
        ax1.plot(epoch_list, train_mean, color=color, linestyle='-', 
                label=f'{agent_name} Train', linewidth=2, alpha=0.8)
        ax1.fill_between(epoch_list, train_mean - train_std, train_mean + train_std, 
                         color=color, alpha=0.2)
        
        # 検証損失のプロット
        ax1.plot(epoch_list, val_mean, color=color, linestyle='--', 
                label=f'{agent_name} Val', linewidth=2, alpha=0.8)
        ax1.fill_between(epoch_list, val_mean - val_std, val_mean + val_std, 
                         color=color, alpha=0.15)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss (Mean ± Std)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 精度のプロット
    for idx, (agent_name, results) in enumerate(all_results.items()):
        color = colors[idx % len(colors)]
        
        # 訓練精度の平均と標準偏差
        train_acc_array = np.array(results['train_acc'])
        train_acc_mean = np.mean(train_acc_array, axis=0)
        train_acc_std = np.std(train_acc_array, axis=0)
        
        # 検証精度の平均と標準偏差
        val_acc_array = np.array(results['val_acc'])
        val_acc_mean = np.mean(val_acc_array, axis=0)
        val_acc_std = np.std(val_acc_array, axis=0)
        
        # 訓練精度のプロット
        ax2.plot(epoch_list, train_acc_mean, color=color, linestyle='-', 
                label=f'{agent_name} Train', linewidth=2, alpha=0.8)
        ax2.fill_between(epoch_list, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std, 
                         color=color, alpha=0.2)
        
        # 検証精度のプロット
        ax2.plot(epoch_list, val_acc_mean, color=color, linestyle='--', 
                label=f'{agent_name} Val', linewidth=2, alpha=0.8)
        ax2.fill_between(epoch_list, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std, 
                         color=color, alpha=0.15)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy (Mean ± Std)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "multi_agent_multi_seed_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n比較プロットを保存しました: {plot_path}")
    
    plt.show()
    
    # 統計情報の表示
    print(f"\n{'='*70}")
    print("=== 最終結果統計 (平均 ± 標準偏差) ===")
    print(f"{'='*70}")
    for agent_name, results in all_results.items():
        train_losses = np.array(results['train_losses'])
        val_losses = np.array(results['val_losses'])
        train_acc = np.array(results['train_acc'])
        val_acc = np.array(results['val_acc'])
        
        print(f"\n{agent_name}:")
        print(f"  シード数: {len(results['seeds'])}")
        print(f"  最終訓練損失: {train_losses[:, -1].mean():.6f} ± {train_losses[:, -1].std():.6f}")
        print(f"  最終検証損失: {val_losses[:, -1].mean():.6f} ± {val_losses[:, -1].std():.6f}")
        print(f"  最終訓練精度: {train_acc[:, -1].mean():.4f} ± {train_acc[:, -1].std():.4f}")
        print(f"  最終検証精度: {val_acc[:, -1].mean():.4f} ± {val_acc[:, -1].std():.4f}")

def save_model_and_results(model, normalizer, train_losses, val_losses, train_acc, val_acc, 
                          agent_name, save_dir, seed):
    """学習済みモデルと結果を保存"""
    model_save_dir = os.path.join(save_dir, f"{agent_name}_model_seed{seed}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    model_path = os.path.join(model_save_dir, f"{agent_name}_model.pth")
    torch.save(model.state_dict(), model_path)
    
    normalizer_path = os.path.join(model_save_dir, f"{agent_name}_normalizer.pkl")
    normalizer.save(normalizer_path)
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_acc,
        'val_accuracies': val_acc,
        'agent_name': agent_name,
        'seed': seed
    }
    
    history_path = os.path.join(model_save_dir, f"{agent_name}_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    return model_save_dir

def main():
    # 固定された設定
    save_dir = "dual_tracking_data"
    
    # Agent1とAgent2の設定
    agents_config = {
        "Agent1": {
            "input_dim": 14,
            "output_dim": 6,
            "hidden_dim": 512,
            "num_layers": 2
        },
        "Agent2": {
            "input_dim": 14,
            "output_dim": 6,
            "hidden_dim": 512,
            "num_layers": 2
        }
    }
    
    epochs = 200
    batch_size = 32
    lr = 1e-3
    num_seeds = 5
    seeds = [42, 123, 456, 789, 1024]  # 使用するシード
    
    print(f"{'='*70}")
    print(f"複数エージェント×複数シード学習")
    print(f"{'='*70}")
    print(f"エージェント数: {len(agents_config)}")
    print(f"各エージェントのシード数: {num_seeds}")
    print(f"総学習回数: {len(agents_config) * num_seeds}")
    print(f"{'='*70}\n")
    
    # 各エージェントのデータセットを読み込み
    agents_datasets = {}
    for agent_name in agents_config.keys():
        train_data, val_data = load_dataset_from_directory(save_dir, agent_name)
        if train_data is None or val_data is None:
            print(f"{agent_name} のデータセット読み込みに失敗しました。")
            return
        agents_datasets[agent_name] = {
            'train': train_data,
            'val': val_data
        }
    
    # 全エージェント×全シードの結果を格納
    all_results = {}
    
    # 各エージェントを学習
    for agent_name, config in agents_config.items():
        print(f"\n{'='*70}")
        print(f"=== {agent_name} の学習開始 ===")
        print(f"{'='*70}")
        
        # 正規化器を作成（訓練データベース）
        normalizer = DataNormalizer()
        normalizer.fit(agents_datasets[agent_name]['train'])
        
        # 結果格納用
        agent_results = {
            'seeds': seeds,
            'train_losses': [],
            'val_losses': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # 各シードで学習
        for seed_idx, seed in enumerate(seeds):
            print(f"\n--- {agent_name} Seed {seed} ({seed_idx+1}/{num_seeds}) ---")
            
            # モデル初期化（シードを変えて）
            model = ForwardDynamicsLSTM(
                input_dim=config['input_dim'],
                output_dim=config['output_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                dropout=0.0,
                seed=seed
            ).to(device)
            
            # 学習実行
            train_losses, val_losses, train_acc, val_acc = train_agent_model(
                model, 
                agents_datasets[agent_name]['train'], 
                agents_datasets[agent_name]['val'],
                normalizer,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                agent_name=agent_name,
                prediction_mode="last_step",
                seed=seed,
                verbose=(seed_idx == 0)  # 最初のシードのみ詳細表示
            )
            
            # 結果保存
            save_model_and_results(
                model, normalizer, train_losses, val_losses, train_acc, val_acc,
                agent_name, save_dir, seed
            )
            
            # 結果を追加
            agent_results['train_losses'].append(train_losses)
            agent_results['val_losses'].append(val_losses)
            agent_results['train_acc'].append(train_acc)
            agent_results['val_acc'].append(val_acc)
            
            print(f"  完了: Train Loss={train_losses[-1]:.6f}, Val Loss={val_losses[-1]:.6f}")
        
        all_results[agent_name] = agent_results
        print(f"\n{agent_name} の全シード学習が完了しました。")
    
    # 比較プロットの作成
    print(f"\n{'='*70}")
    print("=== 全エージェント×全シードの比較プロット作成 ===")
    print(f"{'='*70}")
    plot_multi_agent_multi_seed_results(all_results, save_dir, epochs)
    
    print(f"\n全ての学習が完了しました！結果は {save_dir} に保存されました。")

if __name__ == "__main__":
    main()