tldr：RMA的概念以及在RMA算法的直觉性理解；

Motor Adaptation是神经科学/运动控制领域的概念，人类运动控制中最基础的学习机制之一。
它指的是：当运动系统遭遇外界扰动时，通过逐步修正运动指令来恢复正常表现的过程，这赋予了人类运动系统对外界扰动的快速调整能力。

在100年多前的棱镜实验直观体现了MA：戴上偏移视野的棱镜后，被试伸手偏移。经过 20-30次尝试后逐渐适应重新命中目标。摘下眼镜后出现反方向偏移（后效应），证明大脑的内部模型真的改变了。

继续调查，MA的研究中有一些关于人类运控的发现，负责运控的小脑内部维护了两个dynamics model：
    Forward model/world model: (输入at，st) → 预测下一状态（st+1）
    Inverse model: (输入st，st+1) → 需要的运动指令（at）
MA的本质是更新这两个model，更新的学习信号来自于“身体sensoring vs desire的误差”，也就是脑中推演的S t+1 hat（from world model）和实际S t+1（from sensoring）的差别，它首先驱动世界模型对 dynamics进行修正，进而使逆向模型据此计算出更准确的运动指令，从而补偿这个error；

RMA 定义的是一个问题——机器人需要像人类一样在亚秒级时间尺度上适应新的运动环境。
当前的仿真器还不算一个性能优异的world model，bunches of假设以及优化迭代让其中的动力学不够真实，因此在其中依靠DR等策略训练出来的机器人小脑对外界的环境适应能力不足；我们当然可以让机器人自己进行在线优化（meta learning/realworld RL），从摔倒中学习走路，但是需要大量rollout且成本高；其解决方案是用前向推理替代在线优化来实现这种适应。
作者把前向模型抽象成了对env中环境动力学参数观测（extrinsics）的辨识，我们如果能在环境中实时的得到诸如摩擦力，电机动态等参数，对模型进行校正后也许可以得到一个不错的小脑；这指向了sysid，但显然我们在运动时不需要显式decode这些参数来做控制（只是如果依赖仿真器，会有显式接口的强制要求才需要decode），全部在latent space下表征是一个自然的想法（是不是有点表征学习的感觉）；因此我们期望一个encoder，能在运动时辨识并且表征extrinsics，配合PI完成对外部自适应的运控；

和小脑一样，通过error来观测是自然的，一段运动历史包含了n个【（st，at），st+1】，其中完整包含了全部的系统dynamics，也自然完全保留了extrinsics；我们将其作为输入给到上述的obs encoder，期望他从中表征出dynamics；但是这个训练没有ground truth，目前唯一来源就是simulator，我们如果可以获取一段labeled rollout traj，就可以用来做监督学习从而得到这个obs encoder；

因此我们需要想办法获取到标签以及对应的representation；作者维护了另外一个encoder用来在sim中表征extrinsics（extrinsics2latent_z），他的责任是理解系统的dynamics,能把输入的extrinsics进行represent并且输入pi,进行sim2real train；当pi完成训练,我们认为表征也是准确且有用的（这个represent使得下游的pi表现最优）,他的latent输出被pi使用并且帮助pi认知了env的dynamics;也就相当于完成了latent representasion的标注；

接下来只需要通过 supervised learning 将 obs encoder 的输出对齐 sim encoder 的表征即可。训练数据的收集方式是：用 obs encoder自身的输出驱动冻结的 π 做 rollout（拿到观测历史作为obs encoder的输入），同时从仿真器读取sim encoder的输出作为 label。

RMA的核心贡献是把部署时的环境自适应从一个在线优化问题（需要真实数据+梯度更新）变成了一个前向推理问题（只需 0.5 秒历史+一次网络推理），代价是训练阶段多了一步 φ 的监督学习。

实际上又是一次补差行为，阅读qi老师的文章发现manip的方法又一次来自足式/人形，从阅读到产出note大概3.5hrs，赚飞了；对于表征学习的想法是最近听那个很火的博客看来的，体会是虽然说工科被区分于自然科学，但是不区分于科学，而科学本身就是自然的，理解AI领域的奠基思想之后，再回到自己的一亩三分地，会有恍然大悟的感觉。