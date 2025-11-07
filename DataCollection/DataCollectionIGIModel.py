import torch
import torch.nn as nn
import random
import json
import pickle
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from pathlib import Path
from Environment.EnvIGIModel import DyadTrackingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataSaver:
    """
    collection data and saving file
    """
    def __init__(self, save_dir="/home/hino/Desktop/PointMassInteraction/TrackingData/IGI_tracking_data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.metadata_file = self.save_dir / "data_metadata.json"
        self.metadata = self.load_metadata()
    
    def load_metadata(self):
        """loading meta data file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    
    def save_metadata(self):
        """saving meta data file"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def generate_filename(self, data_type, agent_type, episodes, timesteps, sequence_length):
        """generating file name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{data_type}_{agent_type}_ep{episodes}_ts{timesteps}_seq{sequence_length}_{timestamp}"
    
    def save_datasets_split(self, dataset1_dict, dataset2_dict, data_type, agent_type,
                            episodes, timesteps, sequence_length):
        """saving datasets with train/val split"""
        base_filename = self.generate_filename(data_type, agent_type, episodes, timesteps, sequence_length)
        
        # Agent1 data
        agent1_filename = f"{base_filename}_agent1.pkl"
        agent1_path = self.save_dir / agent1_filename
        with open(agent1_path, 'wb') as f:
            pickle.dump(dataset1_dict, f)  # ← 修正: dataset1_dict を保存
        
        # Agent2 data
        agent2_filename = f"{base_filename}_agent2.pkl"
        agent2_path = self.save_dir / agent2_filename
        with open(agent2_path, 'wb') as f:
            pickle.dump(dataset2_dict, f)  # ← 修正: dataset2_dict を保存
        
        # update metadata
        metadata_key = base_filename
        self.metadata[metadata_key] = {
            'data_type': data_type,
            'agent_type': agent_type,
            'episodes': episodes,
            'timesteps': timesteps,
            'sequence_length': sequence_length,
            'agent1_train_size': len(dataset1_dict['training']),
            'agent1_val_size': len(dataset1_dict['validation']),
            'agent2_train_size': len(dataset2_dict['training']),
            'agent2_val_size': len(dataset2_dict['validation']),
            'agent1_file': agent1_filename,
            'agent2_file': agent2_filename,
            'created_at': datetime.now().isoformat(),
            'input_dim': dataset1_dict['training'][0]['input'].shape[1] if len(dataset1_dict['training']) > 0 else None,  # ← 修正
            'output_dim': dataset1_dict['training'][0]['output'].shape[1] if len(dataset1_dict['training']) > 0 else None  # ← 修正
        }
        self.save_metadata()
        
        print(f"Data has been saved:")
        print(f"  Agent1: {agent1_filename}")
        print(f"    Training: {len(dataset1_dict['training'])} samples")
        print(f"    Validation: {len(dataset1_dict['validation'])} samples")
        print(f"  Agent2: {agent2_filename}")
        print(f"    Training: {len(dataset2_dict['training'])} samples")
        print(f"    Validation: {len(dataset2_dict['validation'])} samples")
        
        return metadata_key
    
    def load_datasets(self, metadata_key):
        """loading datasets"""
        if metadata_key not in self.metadata:
            raise FileNotFoundError(f"Metadata key not found: {metadata_key}")
        
        meta = self.metadata[metadata_key]
        
        # loading agent1 data
        agent1_path = self.save_dir / meta['agent1_file']
        with open(agent1_path, 'rb') as f:
            dataset1 = pickle.load(f)
        
        # loading agent2 data
        agent2_path = self.save_dir / meta['agent2_file']
        with open(agent2_path, 'rb') as f:
            dataset2 = pickle.load(f)
        
        print(f"data has been loaded: {metadata_key}")
        if isinstance(dataset1, dict):
            print(f"  Agent1: 訓練={len(dataset1.get('training', []))} samples, 検証={len(dataset1.get('validation', []))} samples")
            print(f"  Agent2: 訓練={len(dataset2.get('training', []))} samples, 検証={len(dataset2.get('validation', []))} samples")
        else:
            print(f"  Agent1: {len(dataset1)} samples")
            print(f"  Agent2: {len(dataset2)} samples")
        
        return dataset1, dataset2
    
    def list_saved_data(self):
        """display data lists"""
        if not self.metadata:
            print("There is no saved data")
            return
        
        print("\n=== The lists of saved data ===")
        print(f"{'Key':<50} {'Type':<15} {'Episodes':<8} {'Timesteps':<9} {'Train/Val':<15} {'Created':<16}")
        print("-" * 120)
        
        for key, info in self.metadata.items():
            created_date = datetime.fromisoformat(info['created_at']).strftime('%m-%d %H:%M')
            
            # ← 修正: train/val sizeを表示
            if 'agent1_train_size' in info:
                train_val = f"{info['agent1_train_size']}/{info['agent1_val_size']}"
            else:
                # 古い形式の場合
                total_size = info.get('agent1_dataset_size', 0) + info.get('agent2_dataset_size', 0)
                train_val = f"{total_size}"
            
            print(f"{key:<50} {info['data_type']:<15} {info['episodes']:<8} "
                  f"{info['timesteps']:<9} {train_val:<15} {created_date:<16}")
    
    def delete_data(self, metadata_key):
        """delete the data"""
        if metadata_key not in self.metadata:
            print(f"Data does not be found: {metadata_key}")
            return
        
        meta = self.metadata[metadata_key]
        
        # delete the file
        agent1_path = self.save_dir / meta['agent1_file']
        agent2_path = self.save_dir / meta['agent2_file']
        
        if agent1_path.exists():
            agent1_path.unlink()
        if agent2_path.exists():
            agent2_path.unlink()
        
        # del from metadata
        del self.metadata[metadata_key]
        self.save_metadata()
        
        print(f"data has been deleted: {metadata_key}")


def collect_episode_data(env, total_timesteps, sequence_length):
    """
    単一エピソードからデータを収集
    
    Args:
        env: 環境インスタンス
        total_timesteps: 収集するタイムステップ数
        sequence_length: シーケンスの長さ
    
    Returns:
        dataset1, dataset2: 収集されたデータセット
    """
    dataset1 = []
    dataset2 = []
    
    agent1_sequence_inputs = []
    agent2_sequence_inputs = []
    agent1_sequence_outputs = []
    agent2_sequence_outputs = []
    
    # each agent's state
    agent1_state = torch.cat([
        env.agent1_pos_error,
        env.agent1_vel_error,
        env.agent1_acc_error
    ])
    agent2_state = torch.cat([
        env.agent2_pos_error,
        env.agent2_vel_error,
        env.agent2_acc_error
    ])
    
    for step in range(total_timesteps):
        # saving current state
        agent1_state_prev = agent1_state.clone()
        agent2_state_prev = agent2_state.clone()
        
        # input data
        agent1_input = torch.cat([
            agent1_state_prev,
            env.agent1_control,
            env.F_interaction + env.agent1_force,
            env.agent1_self_obs
        ])
        agent2_input = torch.cat([
            agent2_state_prev,
            env.agent2_control,
            -env.F_interaction + env.agent2_force,
            env.agent2_self_obs
        ])
        
        # step environment
        env.step()
        
        # agent's state
        agent1_state = torch.cat([
            env.agent1_pos - env.target_pos,
            env.agent1_vel - env.target_vel,
            env.agent1_acc - env.target_acc
        ])
        agent2_state = torch.cat([
            env.agent2_pos - env.target_pos,
            env.agent2_vel - env.target_vel,
            env.agent2_acc - env.target_acc
        ])
        
        # output data (state change)
        agent1_output = agent1_state - agent1_state_prev
        agent2_output = agent2_state - agent2_state_prev
        
        # add sequence buffer
        agent1_sequence_inputs.append(agent1_input.clone())
        agent2_sequence_inputs.append(agent2_input.clone())
        agent1_sequence_outputs.append(agent1_output.clone())
        agent2_sequence_outputs.append(agent2_output.clone())
        
        # append dataset if dataset length reached
        if len(agent1_sequence_inputs) == sequence_length:
            dataset1.append({
                'input': torch.stack(agent1_sequence_inputs),
                'output': torch.stack(agent1_sequence_outputs)
            })
            dataset2.append({
                'input': torch.stack(agent2_sequence_inputs),
                'output': torch.stack(agent2_sequence_outputs)
            })
            agent1_sequence_inputs.pop(0)
            agent2_sequence_inputs.pop(0)
            agent1_sequence_outputs.pop(0)
            agent2_sequence_outputs.pop(0)
        
        # display progression
        if step % 500 == 0:
            print(f"    Step: {step:,}/{total_timesteps:,} ({step/total_timesteps*100:.1f}%)")
    
    return dataset1, dataset2


def collect_multiple_episodes(
    num_episodes=50,
    total_timesteps=6000,
    crop_timesteps=3000,
    sequence_length=10,
    num_val_episodes=10,
    warmup_steps=200,
    auto_save=True,
    save_dir="/home/hino/Desktop/PointMassInteraction/TrackingData/IGI_tracking_data",
    random_seed=42
):
    """
    複数エピソードでデータ収集し、各エピソードからランダムに切り取る
    
    Args:
        num_episodes: エピソード数
        total_timesteps: 各エピソードで収集する総タイムステップ数
        crop_timesteps: 各エピソードから切り取るタイムステップ数
        sequence_length: シーケンスの長さ
        num_val_episodes: 検証用エピソード数（最後のN個）
        warmup_steps: カルマンフィルタのウォームアップステップ数
        auto_save: 自動保存するか
        save_dir: 保存先ディレクトリ
        random_seed: ランダムシード
    """
    random.seed(random_seed)
    
    train_dataset1 = []
    train_dataset2 = []
    val_dataset1 = []
    val_dataset2 = []
    
    print(f"=== データ収集設定 ===")
    print(f"エピソード数: {num_episodes}")
    print(f"各エピソードの総タイムステップ: {total_timesteps}")
    print(f"各エピソードから切り取るタイムステップ: {crop_timesteps}")
    print(f"ウォームアップステップ: {warmup_steps}")
    print(f"訓練用エピソード: {num_episodes - num_val_episodes}")
    print(f"検証用エピソード: {num_val_episodes}")
    print(f"シーケンス長: {sequence_length}")
    print()
    
    # 切り取り可能な範囲を計算
    max_start = total_timesteps - crop_timesteps
    if warmup_steps > max_start:
        raise ValueError(f"ウォームアップステップ({warmup_steps})が大きすぎます。max_start={max_start}")
    
    num_train_episodes = num_episodes - num_val_episodes
    
    if auto_save:
        saver = DataSaver(save_dir)
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"エピソード {episode+1}/{num_episodes}")
        
        # initialize environment
        env = DyadTrackingEnv(dt=0.01)
        
        print(f"  データ収集中: {total_timesteps} timesteps...")
        episode_dataset1, episode_dataset2 = collect_episode_data(
            env, total_timesteps, sequence_length
        )
        
        print(f"  収集完了: Agent1={len(episode_dataset1)}, Agent2={len(episode_dataset2)} sequences")
        
        # ランダムな開始位置を決定（ウォームアップ後から）
        start_idx = random.randint(warmup_steps, max_start)
        end_idx = start_idx + crop_timesteps - sequence_length + 1  # サンプル数に変換
        
        # データを切り取り
        cropped_dataset1 = episode_dataset1[start_idx:end_idx]
        cropped_dataset2 = episode_dataset2[start_idx:end_idx]
        
        print(f"  切り取り: タイムステップ {start_idx}～{start_idx + crop_timesteps}")
        print(f"  切り取り後: Agent1={len(cropped_dataset1)}, Agent2={len(cropped_dataset2)} sequences")
        
        # 訓練用か検証用かを振り分け
        if episode < num_train_episodes:
            train_dataset1.extend(cropped_dataset1)
            train_dataset2.extend(cropped_dataset2)
            print(f"  → 訓練用データに追加")
        else:
            val_dataset1.extend(cropped_dataset1)
            val_dataset2.extend(cropped_dataset2)
            print(f"  → 検証用データに追加")
    
    print(f"\n{'='*60}")
    print(f"全エピソード完了！")
    print(f"\n最終データサイズ:")
    print(f"  訓練 - Agent1: {len(train_dataset1)}, Agent2: {len(train_dataset2)}")
    print(f"  検証 - Agent1: {len(val_dataset1)}, Agent2: {len(val_dataset2)}")
    
    # 辞書形式でまとめる
    dataset1 = {
        'training': train_dataset1,
        'validation': val_dataset1
    }
    dataset2 = {
        'training': train_dataset2,
        'validation': val_dataset2
    }
    
    # 保存
    saved_key = None
    if auto_save:
        saved_key = saver.save_datasets_split(
            dataset1, dataset2,
            'IGI_tracking', 'random_crop',
            num_episodes, total_timesteps, sequence_length
        )
    
    return dataset1, dataset2, saved_key  # ← 修正: all_dataset1ではなくdataset1


def main():
    """
    main function - data collection demo
    """
    print("=== IGI Tracking Data Collection ===")
    
    # DataSaver
    saver = DataSaver()
    
    # List saved data
    saver.list_saved_data()
    
    print("\nデータ収集オプション:")
    print("1. ランダム切り取り方式（推奨）")
    print("   - 50エピソード × 6000ステップ収集")
    print("   - 各エピソードから3000ステップをランダムに切り取り")
    print("   - 最後の10エピソードを検証用")
    print("2. カスタム設定")
    print("3. 既存データの読み込みテスト")
    print("4. 終了")
    
    while True:
        try:
            choice = input("\n選択してください (1-4): ").strip()
            
            if choice == '1':
                print("\nランダム切り取り方式でデータ収集を開始...")
                dataset1, dataset2, key = collect_multiple_episodes(
                    num_episodes=50,
                    total_timesteps=6000,
                    crop_timesteps=3000,
                    sequence_length=10,
                    num_val_episodes=10,
                    warmup_steps=200,
                    auto_save=True
                )
                print(f"\n保存キー: {key}")
                break
            
            elif choice == '2':
                print("\nカスタム設定:")
                num_ep = int(input("  エピソード数: "))
                total_ts = int(input("  各エピソードの総タイムステップ: "))
                crop_ts = int(input("  切り取るタイムステップ: "))
                val_ep = int(input("  検証用エピソード数: "))
                seq_len = int(input("  シーケンス長: "))
                warmup = int(input("  ウォームアップステップ: "))
                
                dataset1, dataset2, key = collect_multiple_episodes(
                    num_episodes=num_ep,
                    total_timesteps=total_ts,
                    crop_timesteps=crop_ts,
                    sequence_length=seq_len,
                    num_val_episodes=val_ep,
                    warmup_steps=warmup,
                    auto_save=True
                )
                print(f"\n保存キー: {key}")
                break
            
            elif choice == '3':
                saver.list_saved_data()
                if saver.metadata:
                    key = input("読み込むデータのキーを入力: ").strip()
                    try:
                        dataset1, dataset2 = saver.load_datasets(key)
                        print(f"読み込み成功！")
                        
                        if isinstance(dataset1, dict) and 'training' in dataset1:
                            if len(dataset1['training']) > 0:
                                print(f"\nサンプルデータ形状:")
                                print(f"  Input: {dataset1['training'][0]['input'].shape}")
                                print(f"  Output: {dataset1['training'][0]['output'].shape}")
                        elif len(dataset1) > 0:
                            print(f"\nサンプルデータ形状:")
                            print(f"  Input: {dataset1[0]['input'].shape}")
                            print(f"  Output: {dataset1[0]['output'].shape}")
                    except Exception as e:
                        print(f"読み込みエラー: {e}")
                        import traceback
                        traceback.print_exc()
                break
            
            elif choice == '4':
                print("終了します。")
                break
            
            else:
                print("無効な選択です。1-4を入力してください。")
        
        except KeyboardInterrupt:
            print("\n\n処理が中断されました。")
            break
        except Exception as e:
            print(f"エラー: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()