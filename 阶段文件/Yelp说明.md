# Raidar复现实验说明

## Yelp
### 这是用来跑通最小pipeline的一个文件夹。关键的脚本和他们对应的输入输出文件如下：
1. **main_fakeyelp_creator.py** 生成 AI 文本数据  读取文件 *yelp_subset.json*，输出*yelp_GPT_concise.json（AI 原文）*；

2. **main_yelp_gpt_rewrite.py** 调用 OpenAI 做重写（会生成 rewrite 结果 json） 它会读取：
* yelp_GPT_concise.json（AI 原文）
* yelp_human.json（human 原文）
然后分别对两个原文调用openai改写并输出：
* rewrite_yelp_GPT_inv.json
* rewrite_yelp_human_inv.json


3. **detect_yelp_inv.py** 用编辑距离特征训练/检测  输入文件就是第二步的两个输出文件

整个的逻辑如下：
1. Human 原文（人类写的评论）
*数据文件：yelp_human.json（你现在仓库里就有）

2. 生成 AI 原文（让模型“写一段类似的评论”）
*脚本：main_fakeyelp_creator.py
*做法：调用 OpenAI（你贴的代码里是 gpt-3.5-turbo），把 human 原文当提示词生成 AI 文本
*输出文件：yelp_GPT_concise.json

3. 对 human & AI 原文分别做“重写（rewrite）”
*脚本：main_yelp_gpt_rewrite.py（或 llama 版本 main_yelp_llama_rewrite.py）
*做法：再调用一次模型，让它把每条文本改写一遍
*输出文件：
    rewrite_yelp_human_inv.json（human 的改写结果）
    rewrite_yelp_GPT_inv.json（AI 的改写结果）
4. 检测/训练：提特征 → 输出分数/指标
*脚本：detect_yelp_inv.py
*输入：两份 rewrite 文件
*做法：计算“原文 vs 改写文”的编辑距离类特征（以及脚本里实现的其它特征），训练分类器并输出 Accuracy/F1 等指标

用我自己的话来理解就是：我一开始有人类文本，然后用脚本调用openai得到ai文本。然后再用脚本调用openai对这两个文本进行重写。在执行第三个脚本进行特征提取得到分数

### 复现该实验做了哪些工作
1. 让detect_yelp_inv.py 可复现 固定种子。
这样在同样的输入文件、同样的代码、同样的环境下，重复运行得到的结果是一致的。因为脚本中涉及到了随机，但计算机中的是伪随机，固定种子就可以做到复现。
* 这脚本里哪些地方用到了随机性？为什么会影响结果？
A) 训练集/测试集划分（train_test_split）
如果脚本做了：

随机打乱样本

随机抽一部分当测试集

那么每次运行抽到的测试样本可能不同 → 指标就会变。

加 random_state=42 的意义：
每次都用同样的随机方式去切分数据。

B) 模型训练过程本身（例如 MLPClassifier）

神经网络/MLP 这类模型通常会有：

随机初始化权重

随机打乱训练样本顺序（mini-batch）

一些内部的随机策略

所以即使数据划分固定了，模型初始化不同也会导致指标浮动。

加 random_state=42 + np.random.seed(42) 的意义：
模型从同样的起点开始训练，结果更稳定。

2. 