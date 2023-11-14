The code contained within this project implements some of the approaches I have developed during my PhD. 

They are part of **Artificial Intelligence** for **IT Operations** and **Software Engineering** (AIOps for short) that is concerned with using AI to improve the operation of the IT systems through AI.

The content of the project is summarized as follows:
1. **Log Analysis** contains the code for the methods for log analysis, covering three anomaly types: semantic, sequential and performance anomaly detection. It is composed of 4 subprojects:

    * *System Agnostic Log Anomaly Detection*; Creates a log-specific language model from the source code of 1000+ Github projects. Then it fine-tunes it to the system of interest using weakly-supervised learning. The result is a language model that can be used to detect anomalies in the logs of the system of interest. 
    * *System Specific Log Anomaly Detection*; Uses weakly supervised learning on data from the system of interest. The weakly supervised training is achieved by log from systems of the same type (e.g., BGL and Thunderbird two popular benchmarks for log anomaly detection). 
    * *Log Parsing*; Implements a parsing approach, that enables to extraction of the static text of the log instructions from its dynamic variables. It relies on the assumption that static parts of the log text are more frequent than dynamic parts. By using Mask Language Modeling (MLM) the static parts are more likely to be predicted correctly than the dynamic parts (i.e., have a high probability given the context of the nearby words).
    * *Sequential Log Anomaly Detection*; Proposes a deep-clustering methods for discrete data. It uses the method to group logs in time windows that
    are similar to one another. The produced time windows are chained in a sequence and used as input in an HMM, which is used to detect anomalies in the log sequence.

2. **Source Code Analysis** (AI for Software Engineering) contains the code for the analysis of the source code. 
    * *Log Instruction Quality Assessment*; Uses a BERT-based model to assess the quality of the log instructions. The model is trained on a dataset of 1000+ Github projects. The model is used to assess the quality of the log instructions of the system of interest. It is available as a GitHub action as well. 

3. **AIOps Testbed** has the code for automatic deployment of OpenStack, an anomaly-injector alongside a web-application that enables to define of different workloads and injection anomalies in the system. *Soon the data will be available as well*.

4. **Metric Analysis** contains auto-encoder networks for anomaly detection in metric data. 
    * **Root Cause Metrics** It combines auto-encoder networks with approaches from causal learning theory to detect the root cause of an anomaly in a metric. The used data is also available. Take note that the considered case is rather simplistic as it involves several hand-picked golden signal metrics where the anomaly is expected to be reflected. 

The prior projects were published into the following papers: 
1. *Log Analysis*
    
    * J. Bogatinovski, G. Madjarov, S. Nedelkoski, J. Cardoso and O. Kao. "Leveraging Log Instructions in Log-based Anomaly Detection". (2022). IEEE International Conference on Services Computing (SCC), Barcelona, Spain, 2022, pp. 321-326, doi: 10.1109/SCC55611.2022.00053 (h5-index 17, Open Access).
    
    * S. Nedelkoski, J. Bogatinovski, A. Acker, J. Cardoso and O. Kao, "Self-Attentive Classification-Based Anomaly Detection in Unstructured Logs", (2020) IEEE International Conference on Data Mining (ICDM), Sorrento, Italy, 2020, pp. 1196-1201, doi: 10.1109/ICDM50108.2020.00148. (**A* ranking**, h5 index 52)
    
    * J. Bogatinovski*, S. Nedelkoski*, A. Acker, J. Cardoso, O. Kao. (2021). “Self-supervised Log Parsing”. In: Machine Learning and Knowledge Discovery in Databases: Applied Data Science Track. ECML PKDD 2020. Lecture Notes in Computer Science, vol 12460. Springer, Cham. https://doi.org/10.1007/978-3-030-67667-4_8. (h5 index 42, Open Access).
    
    * J. Bogatinovski, S. Nedelkoski, L. Wu, J. Cardoso and O. Kao, "Failure Identification from Unstable Log Data using Deep Learning". (2022). 22nd IEEE International Symposium on Cluster, Cloud and Internet Computing (CCGrid), Taormina, Italy, 2022, pp. 346-355, doi: 10.1109/CCGrid54584.2022.00044 (h-index 24, Open Access).

2. *Source Code Analysis"
    * J. Bogatinovski and O. Kao, "Auto-Logging: AI-centred Logging Instrumentation". (2023). IEEE/ACM 45th International Conference on Software Engineering: New Ideas and Emerging Results (ICSE-NIER), Melbourne, Australia, 2023, pp. 95-100, doi: 10.1109/ICSE-NIER58687.2023.00023 (**A\* ranking** h5 index 85). 

    * J. Bogatinovski, S. Nedelkoski, A. Acker, J. Cardoso, and O. Kao. “QuLog: data-driven approach for log instruction quality assessment”. (2022). In Proceedings of the 30th IEEE/ACM International Conference on Program Comprehension (ICPC '22). Association for Computing Machinery, New York, NY, USA, 275–286. https://doi.org/10.1145/3524610.3527906 (h-index 31, Open Access)

3. *AIOps Testbed*
    * Yet to be published

4. *Metric Analysis*
    * L. Wu, J. Bogatinovski, S. Nedelkoski, J. Tordsson, O. Kao. (2021). “Performance Diagnosis in Cloud Microservices Using Deep Learning”. In: Hacid, H., et al. Service-Oriented Computing – ICSOC 2020 Workshops. ICSOC 2020. Lecture Notes in Computer Science, vol 12632. Springer, Cham. https://doi.org/10.1007/978-3-030-76352-7_13 (h5 index 22, Open Access).

The technical goals I have considered are related to reducing the Mean time to diagnose (MTTD) and mean time to repair (MTTR) of an IT incident.
The main goal of this project is to provide a set of tools that can be used to fasten the diagnosis of the root cause of an IT incident.