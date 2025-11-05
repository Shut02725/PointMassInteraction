import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import sys

# 環境のインポート
from EnvTest import DyadTrackingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_tracking_environment():
    """環境のテスト実行"""
    
    # ========== 1. 環境の初期化 ==========
    print("=" * 60)
    print("DyadTrackingEnv テスト開始")
    print("=" * 60)
    print(f"使用デバイス: {device}")
    
    # ノイズ設定
    noise_config = {
        'visual_pos_noise_std': 0.003,     # 視覚位置ノイズ
        'visual_vel_noise_std': 0.03,      # 視覚速度ノイズ
        'visual_acc_noise_std': 0.3,       # 視覚加速度ノイズ
        'haptic_weber_ratio': 0.10,        # 触覚ウェーバー比 10%
        'haptic_base_noise': 0.1,          # 触覚ベースノイズ 0.1N
        'motor_weber_ratio': 0.05,         # 運動ウェーバー比 5%
        'motor_base_noise': 0.03,          # 運動ベースノイズ 0.03N
        'enable_noise': True               # ノイズ有効化
    }
    
    env = DyadTrackingEnv(dt=0.01, noise_config=noise_config)
    
    # ========== 2. シミュレーション実行 ==========
    num_steps = 3000  # 30秒（dt=0.01なら）
    print(f"\nシミュレーション実行中... ({num_steps}ステップ)")
    
    # 記録用リスト
    errors_agent1 = []
    errors_agent2 = []
    forces_interaction = []
    
    for step in range(num_steps):
        env.step()
        
        # エラーの記録
        error1 = torch.norm(env.agent1_pos - env.target_pos).item()
        error2 = torch.norm(env.agent2_pos - env.target_pos).item()
        force_mag = torch.norm(env.F_interaction).item()
        
        errors_agent1.append(error1)
        errors_agent2.append(error2)
        forces_interaction.append(force_mag)
        
        # 進捗表示
        if (step + 1) % 500 == 0:
            print(f"  ステップ {step + 1}/{num_steps} 完了 "
                  f"(Agent1 error: {error1:.4f}m, Agent2 error: {error2:.4f}m)")
    
    print("シミュレーション完了！")
    
    # ========== 3. パフォーマンス指標の計算 ==========
    print("\n" + "=" * 60)
    print("パフォーマンス指標")
    print("=" * 60)
    
    # 最初の500ステップは除外（過渡応答）
    steady_start = 500
    
    mean_error1 = np.mean(errors_agent1[steady_start:])
    mean_error2 = np.mean(errors_agent2[steady_start:])
    std_error1 = np.std(errors_agent1[steady_start:])
    std_error2 = np.std(errors_agent2[steady_start:])
    mean_force = np.mean(forces_interaction[steady_start:])
    
    print(f"\nAgent 1:")
    print(f"  平均誤差: {mean_error1:.6f} m")
    print(f"  誤差標準偏差: {std_error1:.6f} m")
    print(f"  最大誤差: {max(errors_agent1[steady_start:]):.6f} m")
    
    print(f"\nAgent 2:")
    print(f"  平均誤差: {mean_error2:.6f} m")
    print(f"  誤差標準偏差: {std_error2:.6f} m")
    print(f"  最大誤差: {max(errors_agent2[steady_start:]):.6f} m")
    
    print(f"\n相互作用力:")
    print(f"  平均力: {mean_force:.4f} N")
    print(f"  最大力: {max(forces_interaction):.4f} N")
    
    # ========== 4. 軌跡のプロット ==========
    print("\n軌跡をプロット中...")
    plot_trajectories(env, errors_agent1, errors_agent2, forces_interaction)
    
    # ========== 5. アニメーション作成（オプション） ==========
    create_animation = input("\nアニメーションを作成しますか？ (y/n): ").lower() == 'y'
    if create_animation:
        print("アニメーション作成中...")
        animate_tracking(env)
    
    print("\n" + "=" * 60)
    print("テスト完了！")
    print("=" * 60)

def plot_trajectories(env, errors_agent1, errors_agent2, forces_interaction):
    """軌跡とエラーのプロット"""
    
    # データの取得
    target_traj = np.array(env.trajectory_history['target'])
    agent1_traj = np.array(env.trajectory_history['agent1'])
    agent2_traj = np.array(env.trajectory_history['agent2'])
    time = np.array(env.trajectory_history['time'])
    
    # 図の作成
    fig = plt.figure(figsize=(16, 10))
    
    # ========== 1. 2D軌跡 ==========
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(target_traj[:, 0], target_traj[:, 1], 'k-', linewidth=2, label='Target', alpha=0.7)
    ax1.plot(agent1_traj[:, 0], agent1_traj[:, 1], 'b-', linewidth=1.5, label='Agent 1', alpha=0.6)
    ax1.plot(agent2_traj[:, 0], agent2_traj[:, 1], 'r-', linewidth=1.5, label='Agent 2', alpha=0.6)
    
    # 開始点と終了点をマーク
    ax1.plot(target_traj[0, 0], target_traj[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(target_traj[-1, 0], target_traj[-1, 1], 'mo', markersize=10, label='End')
    
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('2D Trajectory', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # ========== 2. X座標の時間変化 ==========
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(time, target_traj[:, 0], 'k-', linewidth=2, label='Target', alpha=0.7)
    ax2.plot(time, agent1_traj[:, 0], 'b-', linewidth=1.5, label='Agent 1', alpha=0.6)
    ax2.plot(time, agent2_traj[:, 0], 'r-', linewidth=1.5, label='Agent 2', alpha=0.6)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('X Position (m)', fontsize=12)
    ax2.set_title('X Position vs Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ========== 3. Y座標の時間変化 ==========
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(time, target_traj[:, 1], 'k-', linewidth=2, label='Target', alpha=0.7)
    ax3.plot(time, agent1_traj[:, 1], 'b-', linewidth=1.5, label='Agent 1', alpha=0.6)
    ax3.plot(time, agent2_traj[:, 1], 'r-', linewidth=1.5, label='Agent 2', alpha=0.6)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Y Position (m)', fontsize=12)
    ax3.set_title('Y Position vs Time', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ========== 4. トラッキングエラー ==========
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(time, errors_agent1, 'b-', linewidth=1.5, label='Agent 1', alpha=0.7)
    ax4.plot(time, errors_agent2, 'r-', linewidth=1.5, label='Agent 2', alpha=0.7)
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Tracking Error (m)', fontsize=12)
    ax4.set_title('Tracking Error vs Time', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # ========== 5. 相互作用力 ==========
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(time, forces_interaction, 'g-', linewidth=1.5, alpha=0.7)
    ax5.set_xlabel('Time (s)', fontsize=12)
    ax5.set_ylabel('Interaction Force (N)', fontsize=12)
    ax5.set_title('Interaction Force vs Time', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # ========== 6. エラーのヒストグラム ==========
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(errors_agent1[500:], bins=50, alpha=0.6, label='Agent 1', color='blue', density=True)
    ax6.hist(errors_agent2[500:], bins=50, alpha=0.6, label='Agent 2', color='red', density=True)
    ax6.set_xlabel('Tracking Error (m)', fontsize=12)
    ax6.set_ylabel('Probability Density', fontsize=12)
    ax6.set_title('Error Distribution (after 5s)', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tracking_results.png', dpi=300, bbox_inches='tight')
    print("  軌跡プロットを 'tracking_results.png' に保存しました")
    plt.show()

def animate_tracking(env, save_animation=True):
    """トラッキングのアニメーション作成"""
    
    # データの取得
    target_traj = np.array(env.trajectory_history['target'])
    agent1_traj = np.array(env.trajectory_history['agent1'])
    agent2_traj = np.array(env.trajectory_history['agent2'])
    
    # スキップフレーム（アニメーションを速くするため）
    skip = 5
    target_traj = target_traj[::skip]
    agent1_traj = agent1_traj[::skip]
    agent2_traj = agent2_traj[::skip]
    
    # 図の設定
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Dyad Tracking Animation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 軌跡の線（徐々に描画）
    line_target, = ax.plot([], [], 'k-', linewidth=2, label='Target', alpha=0.5)
    line_agent1, = ax.plot([], [], 'b-', linewidth=1.5, label='Agent 1', alpha=0.5)
    line_agent2, = ax.plot([], [], 'r-', linewidth=1.5, label='Agent 2', alpha=0.5)
    
    # 現在位置のマーカー
    marker_target = Circle((0, 0), 0.005, color='black', zorder=10)
    marker_agent1 = Circle((0, 0), 0.005, color='blue', zorder=10)
    marker_agent2 = Circle((0, 0), 0.005, color='red', zorder=10)
    
    ax.add_patch(marker_target)
    ax.add_patch(marker_agent1)
    ax.add_patch(marker_agent2)
    
    # 相互作用を示す線
    line_interaction, = ax.plot([], [], 'g--', linewidth=2, alpha=0.5, label='Interaction')
    
    # 時間表示
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(fontsize=10, loc='upper right')
    
    # トレイル長（過去の軌跡を表示する長さ）
    trail_length = 100
    
    def init():
        line_target.set_data([], [])
        line_agent1.set_data([], [])
        line_agent2.set_data([], [])
        line_interaction.set_data([], [])
        marker_target.center = (0, 0)
        marker_agent1.center = (0, 0)
        marker_agent2.center = (0, 0)
        time_text.set_text('')
        return (line_target, line_agent1, line_agent2, line_interaction,
                marker_target, marker_agent1, marker_agent2, time_text)
    
    def animate(frame):
        # トレイルの開始インデックス
        start_idx = max(0, frame - trail_length)
        
        # 軌跡の更新
        line_target.set_data(target_traj[start_idx:frame+1, 0], 
                            target_traj[start_idx:frame+1, 1])
        line_agent1.set_data(agent1_traj[start_idx:frame+1, 0], 
                            agent1_traj[start_idx:frame+1, 1])
        line_agent2.set_data(agent2_traj[start_idx:frame+1, 0], 
                            agent2_traj[start_idx:frame+1, 1])
        
        # 現在位置のマーカー更新
        marker_target.center = (target_traj[frame, 0], target_traj[frame, 1])
        marker_agent1.center = (agent1_traj[frame, 0], agent1_traj[frame, 1])
        marker_agent2.center = (agent2_traj[frame, 0], agent2_traj[frame, 1])
        
        # 相互作用の線
        line_interaction.set_data([agent1_traj[frame, 0], agent2_traj[frame, 0]],
                                 [agent1_traj[frame, 1], agent2_traj[frame, 1]])
        
        # 時間表示
        time_text.set_text(f'Time: {frame * skip * 0.01:.2f} s')
        
        return (line_target, line_agent1, line_agent2, line_interaction,
                marker_target, marker_agent1, marker_agent2, time_text)
    
    # アニメーション作成
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(target_traj), interval=20,
                                  blit=True, repeat=True)
    
    if save_animation:
        print("  アニメーションを保存中... (時間がかかる場合があります)")
        try:
            anim.save('tracking_animation.gif', writer='pillow', fps=30, dpi=100)
            print("  アニメーションを 'tracking_animation.gif' に保存しました")
        except Exception as e:
            print(f"  アニメーションの保存に失敗しました: {e}")
            print("  アニメーションは画面に表示のみされます")
    
    plt.show()

def compare_with_without_noise():
    """ノイズあり/なしの比較テスト"""
    print("\n" + "=" * 60)
    print("ノイズあり/なしの比較テスト")
    print("=" * 60)
    
    # ノイズなし
    noise_config_off = {
        'visual_pos_noise_std': 0.0,
        'visual_vel_noise_std': 0.0,
        'visual_acc_noise_std': 0.0,
        'haptic_weber_ratio': 0.0,
        'haptic_base_noise': 0.0,
        'motor_weber_ratio': 0.0,
        'motor_base_noise': 0.0,
        'enable_noise': False
    }
    
    # ノイズあり
    noise_config_on = {
        'visual_pos_noise_std': 0.003,
        'visual_vel_noise_std': 0.03,
        'visual_acc_noise_std': 0.3,
        'haptic_weber_ratio': 0.10,
        'haptic_base_noise': 0.1,
        'motor_weber_ratio': 0.05,
        'motor_base_noise': 0.03,
        'enable_noise': True
    }
    
    results = {}
    
    for name, config in [("ノイズなし", noise_config_off), ("ノイズあり", noise_config_on)]:
        print(f"\n{name}でシミュレーション実行中...")
        env = DyadTrackingEnv(dt=0.01, noise_config=config)
        
        errors = []
        for _ in range(2000):
            env.step()
            error1 = torch.norm(env.agent1_pos - env.target_pos).item()
            errors.append(error1)
        
        mean_error = np.mean(errors[500:])
        results[name] = mean_error
        print(f"  平均誤差: {mean_error:.6f} m")
    
    print(f"\n誤差の増加: {(results['ノイズあり'] / results['ノイズなし'] - 1) * 100:.2f}%")

if __name__ == "__main__":
    # メインテスト
    test_tracking_environment()
    
    # 追加テスト（オプション）
    run_comparison = input("\nノイズあり/なしの比較テストを実行しますか？ (y/n): ").lower() == 'y'
    if run_comparison:
        compare_with_without_noise()