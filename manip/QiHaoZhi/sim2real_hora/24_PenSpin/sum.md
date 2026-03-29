知识蒸馏的办法选择

  问题出现在做难度比较大的 manip 任务时，大部分时候只能在仿真器中通过 RL 得到 oracle policy（需要 Reward Engineering 设计奖励函数，通常配合 Privileged Learning 降低学习难度），之后需要 distil 到 deployable policy。

  蒸馏的两种主要方法是 DAgger 和 BC。
  首先我们可以用oracle policy rollout获得其策略分布下的state-action的真值对，但由于轨迹数据具有时序因果关系（非 i.i.d.样本之间互相依赖。st+1被（st，at）影响，而不是图片标注每次都独立同分布），直接使用监督学习（BC）会有compounding error；
  
  因此首选 DAgger——student自己 rollout 整段轨迹，在student 实际到达的每个偏移状态上实时查询 teacher网络得到标签，teacher对整段轨迹进行逐个标注，因此能获得到在早期偏离oracle很多的state下带有纠错性质的aciton label，相当于oracle policy能将对轨迹的纠错能力传递给student，可以缓解因果传递带来的误差累积。
  
  但当任务容错性低时 DAgger 会失败——student 初始 rollout崩溃，episode 极短，收集到的全是不可恢复状态的垃圾数据，在这些状态上物理上不存在合理的纠正动作，因此teacher无法提供有效学习信号。

  DAgger 不可行时退回 BC 作为 fallback。BC 的固有问题是：天然使用监督学习，存在 approximation
  error，再加上轨迹的时序因果关系，导致compounding error，最终在部署时产生distribution shift，模型面对 OOD 输入无法给出合理输出。但 BC 至少能学到运动先验——虽然有漂移，但运动模式的知识被保留了，可以通过后续阶段（如真实数据finetune）来补偿 shift。

  总结二者差别：二者实际上都是监督学习，但是input data/dist 不一样导致效果不一样