# 模型下载
from modelscope import snapshot_download

model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision="v1.0.0", cache_dir=f"./THUDM/chatglm3-6b")

# 模型下载
#
# model_dir = snapshot_download('AI-ModelScope/vicuna-7b-v1.5', cache_dir=f"./vicuna-7b-v1.5")

# 模型下载
# model_dir = snapshot_download('AI-ModelScope/Mistral-7B-Instruct-v0.2', cache_dir=f"./mistral")
