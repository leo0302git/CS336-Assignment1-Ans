# CS336-Assignment1-Ans
本仓库给出针对斯坦福CS336 2025课程作业1的完成代码。

**使用方法**

本仓库代码可以通过所有的`uv run xxx `检查点，并在tinystories数据集上得到较好的训练效果。

- 通过所有的 `uv run xxx `检查点
  - 首先下载/克隆作业项目仓库。作业项目地址：[stanford-cs336/assignment1-basics: Student version of Assignment 1 for Stanford CS336 - Language Modeling From Scratch](https://github.com/stanford-cs336/assignment1-basics/tree/main)
  - 初始化uv虚拟环境，按照作业要求下好对应版本的库
  - 将本仓库中 `cs336_basics` 和 `tests` 中对应的文件粘贴到作业项目的对应文件夹中
  - Powershell（windows系统）进入该仓库。此时应该在 `X:\your\path\to\assignment1-basics`目录下
  - 在该目录下运行 `uv run.txt`中的所有命令，应该都能通过
- 进行训练
  - 训练入口在training.py中
  - 由于数据集很大，所以本仓库不提供数据集。需要您提前准备好，并进行分词化，保存到一个路径下，然后修改代码中关于路径的部分
  - 使用了WandB进行训练过程记录。如果您不需要，可删除相关内容；否则，需要配置好您的WandB账号和API
  - 本仓库使用training.py可以在tinystories数据集上将验证集loss降到1.38，满足作业要求

**注意**

- 使用前需要先配好虚拟环境。另外，本项目可能引入了另外的一些库，导致运行失败。可根据报错信息安装对应库（比如Matplotlib等）
- 代码中涉及路径的内容需要根据您的项目结构进行更改
- 涉及数据集的内容需要提前准备好
- 本作业内容<u>没有</u>完成消融实验，也没有在OpenWebText数据集上进行训练（TODO）

下面是运行 `uv run.txt`中的所有命令，通过所有检查点后的截图

![image-20251128191744788](C:\Users\Leo\Desktop\111\README.assets\image-20251128191744788.png)

下面是在tinystories数据集上的训练结果

![image-20251128191936690](C:\Users\Leo\Desktop\111\README.assets\image-20251128191936690.png)
