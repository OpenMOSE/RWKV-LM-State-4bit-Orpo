import torch

# チェックポイントファイルのパス
checkpoint_path = "base_model/rwkv-x060-14b-world-v2.1-26%trained-20240501-ctx4k.pth"

# チェックポイントファイルを読み込む
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# モデルのstate_dictを取得
model_state_dict = checkpoint#['model_state_dict']

# 各パラメータの名前とサイズを表示
for name, param in model_state_dict.items():
    print(f"Parameter: {name}, Size: {param.size()}")