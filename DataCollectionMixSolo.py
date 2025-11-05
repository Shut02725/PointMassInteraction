import torch
import torch.nn as nn
import random
import pickle
import os
import json
from datetime import datetime
from pathlib import Path

from EnvMixSoloTraining import SoloTrackingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataSaver:
    """
    データ収集とファイル保存を管理するクラス
    """
    def __init__(self, save_dir="./mix_tracking_data_solo"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.metadata_file = self.save_dir / "data_metadata.json"
        self.metadata = self.load_metadata()
    
    def load_metadata(self):
        """メタデータファイルを読み込み"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    
    def save_metadata(self):
        """メタデータファイルを保存"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def generate_filename(self, data_type, agent_type, episodes, timesteps, sequence_length):
        """ファイル名を生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{data_type}_{agent_type}_ep{episodes}_ts{timesteps}_seq{sequence_length}_{timestamp}"
    
    def save_datasets(self, dataset1, dataset2, data_type, agent_type, episodes, timesteps, sequence_length):
        """データセットを保存"""
        # ファイル名生成
        base_filename = self.generate_filename(data_type, agent_type, episodes, timesteps, sequence_length)
        
        # Agent1データの保存
        agent1_filename = f"{base_filename}_agent1.pkl"
        agent1_path = self.save_dir / agent1_filename
        with open(agent1_path, 'wb') as f:
            pickle.dump(dataset1, f)
        
        # Agent2データの保存
        agent2_filename = f"{base_filename}_agent2.pkl"
        agent2_path = self.save_dir / agent2_filename
        with open(agent2_path, 'wb') as f:
            pickle.dump(dataset2, f)
        
        # メタデータの更新
        metadata_key = base_filename
        self.metadata[metadata_key] = {
            'data_type': data_type,
            'agent_type': agent_type,
            'episodes': episodes,
            'timesteps': timesteps,
            'sequence_length': sequence_length,
            'agent1_dataset_size': len(dataset1),
            'agent2_dataset_size': len(dataset2),
            'agent1_file': agent1_filename,
            'agent2_file': agent2_filename,
            'created_at': datetime.now().isoformat(),
            'input_dim': dataset1[0]['input'].shape[1] if len(dataset1) > 0 else None,
            'output_dim': dataset1[0]['output'].shape[1] if len(dataset1) > 0 else None
        }
        
        self.save_metadata()
        
        print(f"データ保存完了:")
        print(f"  Agent1: {agent1_filename} (サイズ: {len(dataset1)})")
        print(f"  Agent2: {agent2_filename} (サイズ: {len(dataset2)})")
        
        return metadata_key
    
    def load_datasets(self, metadata_key):
        """データセットを読み込み"""
        if metadata_key not in self.metadata:
            raise FileNotFoundError(f"Metadata key not found: {metadata_key}")
        
        meta = self.metadata[metadata_key]
        
        # Agent1データの読み込み
        agent1_path = self.save_dir / meta['agent1_file']
        with open(agent1_path, 'rb') as f:
            dataset1 = pickle.load(f)
        
        # Agent2データの読み込み
        agent2_path = self.save_dir / meta['agent2_file']
        with open(agent2_path, 'rb') as f:
            dataset2 = pickle.load(f)
        
        print(f"データ読み込み完了: {metadata_key}")
        print(f"  Agent1: {len(dataset1)} samples")
        print(f"  Agent2: {len(dataset2)} samples")
        
        return dataset1, dataset2
    
    def list_saved_data(self):
        """保存されているデータの一覧を表示"""
        if not self.metadata:
            print("保存されているデータがありません。")
            return
        
        print("\n=== 保存されているデータ一覧 ===")
        print(f"{'Key':<50} {'Type':<15} {'Episodes':<8} {'Timesteps':<9} {'Seq Len':<7} {'Size':<10} {'Created':<16}")
        print("-" * 120)
        
        for key, info in self.metadata.items():
            created_date = datetime.fromisoformat(info['created_at']).strftime('%m-%d %H:%M')
            total_size = info['agent1_dataset_size'] + info['agent2_dataset_size']
            print(f"{key:<50} {info['data_type']:<15} {info['episodes']:<8} "
                  f"{info['timesteps']:<9} {info['sequence_length']:<7} {total_size:<10} {created_date:<16}")
    
    def delete_data(self, metadata_key):
        """データを削除"""
        if metadata_key not in self.metadata:
            print(f"データが見つかりません: {metadata_key}")
            return
        
        meta = self.metadata[metadata_key]
        
        # ファイルを削除
        agent1_path = self.save_dir / meta['agent1_file']
        agent2_path = self.save_dir / meta['agent2_file']
        
        if agent1_path.exists():
            agent1_path.unlink()
        if agent2_path.exists():
            agent2_path.unlink()
        
        # メタデータから削除
        del self.metadata[metadata_key]
        self.save_metadata()
        
        print(f"データ削除完了: {metadata_key}")

def collect_solo_tracking_data(total_timesteps=10000, sequence_length=10, auto_save=True, save_dir="./solo_tracking_data"):
    """
    デュアルトラッキング環境からシーケンシャルデータを収集する関数
    
    Args:
        total_timesteps: 総ステップ数
        sequence_length: シーケンス長
        auto_save: 自動保存するかどうか
        save_dir: 保存先ディレクトリ
    
    Returns:
        dataset1, dataset2, saved_key: Agent1とAgent2のデータセットと保存キー
    """
    dataset1 = []
    dataset2 = []
    print(f"時系列データ収集開始: {total_timesteps}ステップ, シーケンス長={sequence_length}")

    if auto_save:
        saver = DataSaver(save_dir)

    env = SoloTrackingEnv(dt=0.01)

    agent1_sequence_inputs = []
    agent2_sequence_inputs = []  # タイポ修正
    agent1_sequence_outputs = []
    agent2_sequence_outputs = []

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
        # 現在の状態を保存（変化量計算用）
        agent1_state_prev = agent1_state.clone()
        agent2_state_prev = agent2_state.clone()

        # 入力データを構築（状態、制御入力、相互作用力、自己観測）
        agent1_input = torch.cat([
            agent1_state_prev,
            env.agent1_control,
            env.F_interaction + env.agent1_force,
            env.agent1_self_obs
        ])

        agent2_input = torch.cat([
            agent2_state_prev,
            env.agent2_control,
            - env.F_interaction + env.agent2_force,  # agent1_force → agent2_force に修正
            env.agent2_self_obs
        ])

        # 環境を1ステップ進める
        env.step()

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

        # 出力データ（状態変化量）を計算
        agent1_output = agent1_state - agent1_state_prev
        agent2_output = agent2_state - agent2_state_prev

        # シーケンスバッファに追加
        agent1_sequence_inputs.append(agent1_input.clone())
        agent2_sequence_inputs.append(agent2_input.clone())
        agent1_sequence_outputs.append(agent1_output.clone())
        agent2_sequence_outputs.append(agent2_output.clone())

        # シーケンス長に達したらデータセットに追加
        if len(agent1_sequence_inputs) == sequence_length:
            dataset1.append({
                'input': torch.stack(agent1_sequence_inputs),  # (sequence_length, input_dim)
                'output': torch.stack(agent1_sequence_outputs)  # (sequence_length, output_dim)
            })
            dataset2.append({
                'input': torch.stack(agent2_sequence_inputs),  # (sequence_length, input_dim)
                'output': torch.stack(agent2_sequence_outputs)  # (sequence_length, output_dim)
            })
            
            agent1_sequence_inputs.pop(0)
            agent2_sequence_inputs.pop(0)
            agent1_sequence_outputs.pop(0)
            agent2_sequence_outputs.pop(0)

        # プログレス表示
        if step % 100 == 0:  # 進捗表示の頻度を調整
            print(f"Step: {step:,}/{total_timesteps:,} ({step/total_timesteps*100:.1f}%) - Collected sequences: {len(dataset1)}")
            
    print(f"データ収集完了！")
    print(f"Agent1データセット: {len(dataset1)} sequences")
    print(f"Agent2データセット: {len(dataset2)} sequences")
    
    # 最終データの自動保存
    saved_key = None
    if auto_save and len(dataset1) > 0:
        saved_key = saver.save_datasets(
            dataset1, dataset2, 
            'solo_tracking', 'collected',  # 'final' → 'collected' に変更
            0, total_timesteps, sequence_length  # total_episodesが定義されていないため0に修正
        )
    
    return dataset1, dataset2, saved_key

def collect_multiple_episodes(num_episodes=100, timesteps_per_episode=2000, sequence_length=10, 
                            auto_save=True, save_dir="./solo_tracking_data"):
    """
    複数エピソードにわたってデータを収集する関数
    
    Args:
        num_episodes: エピソード数
        timesteps_per_episode: エピソードあたりのステップ数
        sequence_length: シーケンス長
        auto_save: 自動保存するかどうか
        save_dir: 保存先ディレクトリ
    """
    all_dataset1 = []
    all_dataset2 = []
    
    print(f"複数エピソードデータ収集開始: {num_episodes}エピソード × {timesteps_per_episode}ステップ")
    
    if auto_save:
        saver = DataSaver(save_dir)
    
    for episode in range(num_episodes):
        print(f"\nエピソード {episode+1}/{num_episodes}")
        
        # 環境を新しく初期化
        env = SoloTrackingEnv(dt=0.01)
        
        episode_dataset1, episode_dataset2, _ = collect_solo_tracking_data(
            total_timesteps=timesteps_per_episode,
            sequence_length=sequence_length,
            auto_save=False  # エピソード毎の保存は無効
        )
        
        all_dataset1.extend(episode_dataset1)
        all_dataset2.extend(episode_dataset2)
        
        print(f"エピソード{episode+1}完了: Agent1={len(episode_dataset1)}, Agent2={len(episode_dataset2)} sequences")
    
    print(f"\n全エピソード完了！")
    print(f"総データ数: Agent1={len(all_dataset1)}, Agent2={len(all_dataset2)} sequences")
    
    # 全データの自動保存
    saved_key = None
    if auto_save and len(all_dataset1) > 0:
        saved_key = saver.save_datasets(
            all_dataset1, all_dataset2,
            'solo_tracking', 'multi_episode',
            num_episodes, num_episodes * timesteps_per_episode, sequence_length
        )
    
    return all_dataset1, all_dataset2, saved_key

def main():
    """
    メイン実行関数 - データ収集のデモ
    """
    print("=== デュアルトラッキングデータ収集システム ===")
    
    # DataSaverのインスタンス作成
    saver = DataSaver()
    
    # 既存データの一覧表示
    saver.list_saved_data()
    
    # データ収集の選択肢
    print("\nデータ収集オプション:")
    print("1. 単一セッション（10,000ステップ）【推奨：多様な軌道】")
    print("2. 複数エピソード（100エピソード × 2,000ステップ）")
    print("3. カスタム設定")
    print("4. 既存データの読み込みテスト")
    print("5. 終了")
    
    while True:
        try:
            choice = input("\n選択してください (1-5): ").strip()
            
            if choice == '1':
                print("単一セッションデータ収集を開始...")
                dataset1, dataset2, key = collect_solo_tracking_data(
                    total_timesteps=10000, 
                    sequence_length=10
                )
                print(f"保存キー: {key}")
                break
                
            elif choice == '2':
                print("複数エピソードデータ収集を開始...")
                dataset1, dataset2, key = collect_multiple_episodes(
                    num_episodes=100,
                    timesteps_per_episode=2000
                )
                print(f"保存キー: {key}")
                break
                
            elif choice == '3':
                timesteps = int(input("総ステップ数を入力: "))
                seq_len = int(input("シーケンス長を入力: "))
                dataset1, dataset2, key = collect_solo_tracking_data(
                    total_timesteps=timesteps,
                    sequence_length=seq_len
                )
                print(f"保存キー: {key}")
                break
                
            elif choice == '4':
                saver.list_saved_data()
                if saver.metadata:
                    key = input("読み込むデータのキーを入力: ").strip()
                    try:
                        dataset1, dataset2 = saver.load_datasets(key)
                        print(f"読み込み成功！")
                        if len(dataset1) > 0:
                            print(f"サンプルデータ形状:")
                            print(f"  Input: {dataset1[0]['input'].shape}")
                            print(f"  Output: {dataset1[0]['output'].shape}")
                    except Exception as e:
                        print(f"読み込みエラー: {e}")
                break
                
            elif choice == '5':
                print("終了します。")
                break
                
            else:
                print("無効な選択です。1-5を入力してください。")
                
        except KeyboardInterrupt:
            print("\n\n処理が中断されました。")
            break

if __name__ == "__main__":
    main()