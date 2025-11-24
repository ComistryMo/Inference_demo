import torch
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

model_name = "./Qwen3-0.6B"

# Helper function to format parameter counts
def format_params(num_params):
    """Converts a number of parameters into a human-readable string (B, M, K)."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f} B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f} M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f} K"
    else:
        return f"{num_params}"
    
def view_model_info(path: str):
    # 1. 加载模型配置
    print("正在加载模型配置...")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)

    # 2. 使用 init_empty_weights 上下文管理器
    # 在这个上下文里创建的模型，其参数将不占用实际内存
    print("正在构建无内存占用的模型骨架...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # 3. 打印模型结构
    print("\n--- 模型结构 (无内存占用) ---")
    print(model) # 打印整个模型结构会非常长，这里注释掉，如果需要可以取消注释

    # 检查一下模型参数所在的设备，你会发现它们都在 'meta' device 上
    print(f"\n模型参数设备: {model.device}")
    print(f"任意一个参数的设备: {next(model.parameters()).device}")

    # --- 新增部分：打印各个部分的参数量 ---
    print("\n--- 各个部分的参数量 ---")

    # 存储每个模块的名称和其包含的参数量
    module_param_counts = []

    # 遍历模型的所有命名模块
    for name, module in model.named_modules():
        # 计算当前模块及其所有子模块的总参数量
        num_params = sum(p.numel() for p in module.parameters())
        if num_params > 0: # 只记录有参数的模块
            module_param_counts.append((name, num_params))

    # 对结果进行排序，通常按模块名称排序可以提高可读性
    module_param_counts.sort(key=lambda x: x[0])

    # 打印格式化的参数量信息
    print(f"{'模块名称':<80} {'参数量':>20}")
    print(f"{'-'*80} {'-'*20}")

    for name, count in module_param_counts:
        print(f"{name:<80} {format_params(count):>20}")

    # 计算并打印整个模型的总参数量
    total_model_params = sum(p.numel() for p in model.parameters())
    print(f"{'-'*80} {'-'*20}")
    print(f"{'总参数量':<80} {format_params(total_model_params):>20}")

    # --- 提取并对比特定部分的参数量 ---
    print("\n--- 主要模块参数量对比 ---")

    vit_params = 0
    merger_params = 0
    language_model_params = 0
    lm_head_params = 0 # lm_head 是独立于 language_model 的输出层

    for name, count in module_param_counts:
        if name == "model":
            language_model_params = count
        elif name == "lm_head":
            lm_head_params = count

    print(f"{'语言模型 (不含lm_head)':<30} {format_params(language_model_params):>15}") # 语言模型模块本身包含embed_tokens和layers，lm_head是单独的输出层
    print(f"{'语言模型输出层 (lm_head)':<30} {format_params(lm_head_params):>15}")
    print(f"{'语言模型 (总和)':<30} {format_params(language_model_params + lm_head_params):>15}") # 语言模型总参数量通常指 language_model + lm_head

def main():
    view_model_info("./Qwen3-0.6B")

if __name__ == "__main__":
    main()