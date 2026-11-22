# 主基线：2 障碍 +100 / 3 障碍 +100

日期：2026-11-22  
模型：MZX-0.1 / 500 → 1000 节点  
生长：5 轮，每轮 +100  
gate：可学习，初始 0.5  
贡献阈值：0.0001  
评估：100 episodes

---

## 1. 2 障碍主基线

基座：mzx_bc_2obs_direct.pt（BC 84%、DAgger 85%）  
任务：2 随机障碍，目标 ±0.6

结果：

~~~text
nodes = 1000
success = 0.84
steps = 121.25
checkpoint = mzx_2obs_plus100.pt
~~~

---

## 2. 3 障碍主基线

基座：mzx_bc_3obs.pt（BC 86%）  
任务：3 固定障碍，目标 ±0.6

结果：

~~~text
nodes = 1000
success = 0.93
steps = 155.73
checkpoint = mzx_3obs_plus100.pt
~~~

---

## 3. 结论

> 两个 +100 主基线都建好了。

- 2 障碍：84%
- 3 障碍：93%

3 障碍这一轮很高，2 障碍这一轮偏低（之前 3 seeds 均值 87.33%），说明 seed 仍有影响。

---

## 4. 下一步

1. 用这两个 checkpoint 做主基线
2. 2 障碍基线建议多 seed 复核
3. 再跑 F–J 高级消融

---

## 5. 一句话

> 2 障碍 +100 主基线 84%，3 障碍 +100 主基线 93%；checkpoint 已保存。

---

## 6. 文件

- build_main_baselines.py
- main_baselines_result.json
- mzx_2obs_plus100.pt
- mzx_3obs_plus100.pt
- MAIN_BASELINES_REPORT.md
