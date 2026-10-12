**RAG 初始语料：来源与适用范围**

本页记录 [knowledge/corpus.json](../knowledge/corpus.json) 的人工选材与归属信息。八篇笔记均于 **2026-09-08** 核对，使用简短中文转述；来源标识和精确 URL 随检索结果保留。没有下载论文全文、数据集、图片或真实参与者记录。

| 来源 ID | 官方来源或原始论文 | 页面标示的日期 | 核对位置与收录用途 |
|---|---|---|---|
| `aha_heart_rate_activity` | [AHA：心率说明](https://www.heart.org/en/health-topics/high-blood-pressure/the-facts-about-high-blood-pressure/all-about-heart-rate-pulse) | 复核：2024-05-13 | 运动与心率、其他影响因素；解释 HR 与活动背景需要一起考虑 |
| `empatica_eda_context` | [Empatica：EDA 说明](https://support.empatica.com/hc/en-us/articles/115000198283-What-is-electrodermal-activity) | 未提供可核对的正文修订日期 | EDA 定义及影响因素；说明皮肤电测量与采集条件 |
| `apple_ppg_sampling` | [Apple：心率记录与测量](https://support.apple.com/en-us/120277) | 发布：2025-11-07 | 背景测量时机及光学原理；说明真实设备采样协议需单独核对 |
| `apple_sensor_limitations` | [Apple：提高测量准确性](https://support.apple.com/en-us/105002) | 发布：2026-03-10 | 心率传感器影响因素及佩戴；说明设备测量限制 |
| `ninds_eeg_measurement` | [NIH/NINDS：神经检查说明](https://www.ninds.nih.gov/health-information/disorders/neurological-diagnostic-tests-and-procedures) | 未核实正文修订日期 | 脑电记录章节；区分真实 EEG 记录与合成代理特征 |
| `cdc_sleep_quality` | [CDC：睡眠说明](https://www.cdc.gov/sleep/about/index.html) | 页面日期：2024-05-15 | 睡眠质量章节；说明质量与时长的概念差异 |
| `fda_general_wellness_scope` | [FDA：一般健康产品指南](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/general-wellness-policy-low-risk-devices) | 指南：2026-01 | 指南落地页的用途范围；提供一般健康信息与医学用途的边界背景 |
| `smets_wearable_context_2018` | [Smets 等：日常压力的可穿戴研究](https://www.nature.com/articles/s41746-018-0074-9) | 发布：2018-12-12 | 摘要、活动与信号质量分析、讨论；说明情境因素与研究适用范围 |

英文原始标题、章节定位和归属说明保存在各文档元数据中。上表日期来自本次读取到的页面，不将搜索引擎抓取时间当成发布日期。NINDS 条目依据搜索索引返回的官方脑电章节文本核对；直接打开正文未成功，因此没有宣称核实其修订日期或完整页面内容。

收录的原始研究是 Smets、Rios Velazquez、Schiavone 等作者的 2018 年论文，DOI 为 `10.1038/s41746-018-0074-9`。出版页面明确标示 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；本语料保留作者、来源及许可，并明确说明已压缩改写为中文。未复制论文图片、表格或参与者数据。其他页面采用有限事实转述和来源链接，没有将“可公开访问”表述为整页内容可任意再发布。

每篇正文的“项目适用边界”是维护者基于 [合成数据生成器](../generate_demo_data.py) 和 [预处理协议](../app/pipeline/preprocess.py) 所写的工程说明。例如，`EEG_feature` 是合成代理量，`sleep_quality` 是会话内固定的 0–1 数值，在线输入按 1 Hz、30 秒窗口处理。这些实现选择没有被 NIH、CDC、设备厂商或论文作者验证。

语料不收录所访问页面中的疾病分类、诊疗建议或个体医学判断。FDA 条目只说明该来源的用途边界，不提供本项目的监管分类结论；设备厂商条目只描述对应文档涉及的设备，不能推广为所有可穿戴设备的性能承诺。研究论文提供背景证据，不是当前合成模型的准确率或临床效果证明。

检索组件可以利用这些材料说明“应核对哪些背景因素”，但不能仅因某段材料相似，就断言“该用户高分由某种病因导致”。文献中的真实生理联系与本次模型的特征贡献是不同层面的信息，最终输出必须保留这种区分。

[manifest.json](../knowledge/manifest.json) 保存本地内容与来源元数据的哈希。它能够发现本地语料发生变化，不能证明远程页面此后没有更新。该初始版本只覆盖上述主题；未定期自动刷新，也不自称为完整或持续保持最新的健康指南。
