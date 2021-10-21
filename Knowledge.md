# Knowledge-aware Trust-native AI

## 1. Principles, Surveys, and Tutorials 
Note: The major concerns in Trustworthy AI include (but not limited to) Explanability, Robustness, Privacy and Security, Fairness.
1. Trustworthy AI: From Principles to Practices [[Paper](https://arxiv.org/pdf/2110.01167.pdf)]
> * This survey provides a good roadmap for the important aspects of AI trustworthiness.
2. Trusted AI 101: A Guide to Building Trustworthy and Ethical AI Systems [[Website](https://www.datarobot.com/trusted-ai-101/)]
3. IJCAI 2020 Tutorial: Trusting AI by Testing and Rating Third Party Offerings [[Website](https://sites.google.com/view/ijcai2020tut-aitrust/home)]
4. KDD 2021 Tutorial: Machine Learning Robustness, Fairness, and their Convergence [[Website (with slides)](https://kdd21tutorial-robust-fair-learning.github.io/)]
5. KDD 2021 Tutorial: Machine Learning Explainability and Robustness: Connected at the Hip [[Website (with slides)](https://sites.google.com/andrew.cmu.edu/kdd-2021-tutorial-expl-robust/home)]
6. KDD 2020 Tutorial: Intelligible and Explainable Machine Learning: Best Practices and Practical Challenges [[Video](https://www.youtube.com/watch?v=gjJIHIIbbok)]
7. Informed Machine Learning – A Taxonomy and Survey of Integrating Prior Knowledge into Learning Systems (TKDE 2021) [[Paper](https://arxiv.org/pdf/1903.12394.pdf)] 🌟
8. Knowledge graph semantic enhancement of input data for improving AI (IEEE Internet Computing 2020)
9. Exploiting knowledge graphs in industrial products and services: A survey of key aspects, challenges, and future perspectives (Computers in Industry 2021)
10. Cognitive Graph for Multi-Hop Reading Comprehension at Scale (ACL 2019) [[GitHub](https://github.com/THUDM/CogQA)] [[Notes 1 (by author in Chinese)](https://zhuanlan.zhihu.com/p/72981392)] [[Notes 2 by a reader in Chinese](https://blog.csdn.net/XiangJiaoJun_/article/details/105879690)]
> * Major contribution：System 1 (from training data) + System 2 (from existing knowledge or rules) to enhance the performance of the downstream tasks. This guides the direction of Trustworthy AI.
> * Note: Cognitive Graph is not directly equal to Knowledge Graph. You can view CG as a (dynamic, partial, local) KG generated instantly from the query.
11. Awesome Explainable AI [[GitHub](https://github.com/wangyongjie-ntu/Awesome-explainable-AI)]
12. Awesome_deep_learning_interpretability [[GitHub](https://github.com/oneTaken/awesome_deep_learning_interpretability)]
13. Awesome-machine-learning-interpretability [[GitHub](https://github.com/jphall663/awesome-machine-learning-interpretability)]
14. Awesome-explainable-interpretable-ai [[Website](https://myrelated.work/t/awesome-explainable-interpretable-ai/40)]
-------------------------------

## 2. Trustworthy AI with Specific Domain Knowledge
### 2.1. Healthcare and Medical Domain
#### Tutorials, Surveys and Workshop
1. KDD 2021 Tutorial: Software as a Medical Device: Regulating AI in Healthcare via Responsible AI [[Paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3470823)] [[Website](https://responsibleml.github.io/)] [[Video](https://www.youtube.com/watch?v=p-Kg27--MII)]
2. AAAI 2021 Workshop: Trustworthy AI for Healthcare [[Website](https://taih20.github.io)]
3. KDD 2020 Workshop on Applied Data Science for Healthcare: Trustable and Actionable AI for Healthcare [[Website](https://dshealthkdd.github.io/dshealth-2020/)]
> * Most of the papers mainly focus on effectineness of the downstream tasks.
4. On Assessing Trustworthy AI in Healthcare. Machine Learning as a Supportive Tool to Recognize Cardiac Arrest in Emergency Calls [[Paper](https://www.frontiersin.org/articles/10.3389/fhumd.2021.673104/full)] 
5. A Survey on Explainable Artificial Intelligence (XAI): Toward Medical XAI (IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS 2020) [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9233366)]
6. What do we need to build explainable AI systems for the medical domain? (arxiv 2017) [[Paper](https://arxiv.org/pdf/1712.09923.pdf)] an old paper...
7. KDD 2020 Tutorial on Human-Centered Explainability for Healthcare [[Website](https://healthxaitutorial.github.io/kdd2020/)]
8. KDD 2021 Tutorial: Advances in Mining Heterogeneous Healthcare Data [[Slides](https://sites.psu.edu/kdd2021tutorial/files/2021/08/KDD21_tutorial.pdf)]
9. Open Data Science Conference West (ODSC) Tutorial: DEEP LEARNING FOR HEALTHCARE [[Website](http://dl4health.org)]

#### Explainability
1. DLIME: A Deterministic Local Interpretable Model-Agnostic Explanations Approach for Computer-Aided Diagnosis Systems (KDD 2019 workshop) [[Paper](https://arxiv.org/pdf/1906.10263.pdf)]  [[GitHub](https://github.com/rehmanzafar/dlime_experiments.git)] 
> * More close to XAI: This work proposes a deterministic version of LIME. Instead of random perturbation, we utilize agglomerative Hierarchical Clustering (HC) to group the training data together and K-Nearest Neighbour (KNN) to select the relevant cluster of the new instance that is being explained. After finding the relevant cluster, a linear model is trained over the selected cluster to generate the explanations.
3. DETERRENT: Knowledge Guided Graph Attention Network for Detecting Healthcare Misinformation (KDD 2020) 🌟 [[Paper](http://pike.psu.edu/publications/kdd20-deterrent.pdf)] [[GitHub](https://github.com/cuilimeng/DETERRENT)] `Healthcare Misinformation Detection`
> * A novel problem of explainable healthcare misinformation detection (from the web) by leveraging medical knowledge graph to better capture the high-order relations between entities.
> * RGCN (with attention) for KG reasoning + text encoer of articles = learn the representation for each earticle, then formulate a classification problem to distinguish if a news is fake.
> * The support KG: KnowLife: a versatile approach for constructing a large knowledge graph for biomedical sciences [[Paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.798.9505&rep=rep1&type=pdf)] [[Website](http://knowlife.mpi-inf.mpg.de)]
> * Similar basic code (text+GRU+RGCN): Learning to Update Knowledge Graphs by Reading News [[GitHub](https://github.com/esddse/GUpdater)]
4. INPREM: An Interpretable and Trustworthy Predictive Model for Healthcare (KDD 2020) 🌟 [[Paper](http://homepage.divms.uiowa.edu/~jrusert/momina_423.pdf)] `Risk Prediction` `EHR`
5. MedPath: Augmenting Health Risk Prediction via Medical Knowledge Paths (WWW 2021) [[GitHub](https://github.com/machinelearning4health/MedPath)] `Risk Prediction` `EHR`
> * Personalized KG to provided personalized prediction and explicit reasoning.
> * The major idea is borrowed from MHGRN (multi-hop graph): Scalable Multi-Hop Relational Reasoning for Knowledge-Aware Question Answering (EMNLP 2020) [[Paper](https://arxiv.org/pdf/2005.00646.pdf)] [[Notes in Chinese](https://blog.csdn.net/ld326/article/details/114049909)]
6. HiTANet: Hierarchical Time-Aware Attention Networks for Risk Prediction on Electronic Health Records (KDD 2020) 🌟 [[GitHub](https://github.com/HiTANet2020/HiTANet)] `Risk Prediction` `EHR`
7. StageNet: Stage-Aware Neural Networks for Health Risk Prediction (WWW 2020) `Risk Prediction`
8. MedRetriever: Target-Driven Health Risk Prediction via Retrieving Unstructured Medical Text (CIKM 2021) `Risk Prediction`
9. Online Disease Diagnosis with Inductive Heterogeneous Graph Convolutional Networks (WWW 2021) `Disease Diagnosis`
10. COMPOSE: Cross-Modal Pseudo-Siamese Network for Patient Trial Matching (KDD 2020) `Patenet Trail Matching` [[GitHub](https://github.com/v1xerunt/COMPOSE)]
> * There is trial data provided. But still needs the EHR. The Github provides a sample format of the required EHR.
12. CORE: Automatic Molecule Optimization using Copy and Refine Strategy (AAAI 2020) `Drug Discovery` [[GitHub](https://github.com/futianfan/CORE)] (seems no data T_T)

#### Fairness and Ethics

#### Privacy and Security
1. Communication Efficient Federated Generalized Tensor Factorization for Collaborative Health Data Analytics (WWW 2021) [[Paper](http://cs.emory.edu/site/aims/pub/ma21www.pdf)]
> * Federated learning offers a privacy-preserving paradigm for collaborative learning among different entities, which seemingly provides an ideal potential to further enhance the tensor factorization-based collaborative phenotyping to handle sensitive personal health data.
> * This paper addresses 3 issues: (1) restrictions to the classic tensor factorization, (2) high communication cost and (3) reduced accuracy, on two real-world electronics health record datasets.

#### Truth Discovery and Knowledge Verification (more close to downstream applications)
1. When Truth Discovery Meets Medical Knowledge Graph: Estimating Trustworthiness Degree for Medical Knowledge Condition (CoRR 2018)
> * Supplement medical knowledge graph with knowledge condition information.
> * The knowledge triples and conditions serve as objects and claims, and each doctor or a user provides answers on QA website is a source. The proposed method has two novel
properties: 1) Combining prior source quality information and automatic source reliability estimation; 2) Encoding the object (knowledge triple) information into the proposed method.
2. SMR: Medical Knowledge Graph Embedding for Safe Medicine Recommendation (Big Data Research 2021) [[Paper] (https://www.sciencedirect.com/science/article/pii/S2214579620300423?casa_token=tVr0i-xSshMAAAAA:TwCct8Fk4IWKv5P3O0pS1rAfmZWTuZDAkbw1a44QUjYsYufxQ7u8wosPbzyULxVJ5nJcNF62Pxo)] `Treatment Recommendation`

#### Related readings on Knowledge-aware Healthcare
1. Advances in Mining Heterogenous Healthcare Data (KDD 2020 Tutotial) [[Slides](https://sites.psu.edu/kdd2021tutorial/files/2021/08/KDD21_tutorial.pdf)]
2. Clinical Trial Parser by Facebook Research [[GitHub](https://github.com/facebookresearch/Clinical-Trial-Parser)]
> * Downstream task: medical NER

#### Talented People to Follow
1. Fenglong Ma [[Website](http://www.personal.psu.edu/ffm5105/Research.html)]
2. Fatma Özcan [[DBLP](https://dblp.org/pid/o/FatmaOzcan.html)]
3. Lei Chuan [[Website](https://leichuan.github.io/publications/)]
4. Walter F Stewart [[Google Scholar](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwj7rZOIxtXzAhVtJaYKHWQZAlgQFnoECAYQAQ&url=https%3A%2F%2Fscholar.google.com%2Fcitations%3Fuser%3DflGRoHEAAAAJ%26hl%3Den&usg=AOvVaw23BN6oa909_j2fm2x7Tgp1)]
5. Cao Xiao [[Website](https://sites.google.com/view/danicaxiao/home)]

#### Datasets
1. PubMed
2. MDX [[Link](https://www.ibm.com/products/micromedex-with-watson)]
3. MIMIC-III [[Reference](https://www.nature.com/articles/sdata201635)] `documented EHR` [[Data download (need to be a credentialed user)](https://mimic.mit.edu/docs/gettingstarted/)]
4. Bio CDR [[Reference](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4860626/)]
5. NCBI [[Reference](https://pubmed.ncbi.nlm.nih.gov/24393765/)], NCBID [[Reference](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/)]
6. ShARe [[Reference](https://aclanthology.org/S14-2007.pdf)]
7. BioCreative [[Reference](https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus/)]
8. Summary from NormCo [[Github](https://github.com/IBM/aihn-ucsd/tree/master/NormCo-deep-disease-normalization)]
9. Datasets provided by [[MedType](https://github.com/svjan5/medtype)]: [[WikiMed](https://drive.google.com/u/0/uc?export=download&confirm=seZN&id=16suJCinjfYhw1u1S-gPFmGFQZD331u7I)] and [[PubMedDS](https://drive.google.com/u/0/uc?export=download&confirm=kB20&id=16mEFpCHhFGuQ7zYRAp2PP3XbAFq9MwoM)]
10. Unified Medical Language System (UMLS): 4.2 million biomedical concepts, with 127 types
> * There is a UMLS Semantic Network for concept mapping to semantic types?
11. MedMetions [[Reference](https://arxiv.org/pdf/1902.09476.pdf)]
12. CBLUE (Chinese NLP Medical Text Mining) [[Link](https://tianchi.aliyun.com/specials/promotion/2021chinesemedicalnlpleaderboardchallenge)]
13. 中文医疗领域自然语言处理相关数据集、经典论文资源蒸馏分享 [[Link](https://mp.weixin.qq.com/s__biz=MzIxNDgzNDg3NQ==&mid=2247489095&idx=1&sn=36889ef5e30293b1e204bd807f83c5d8&chksm=97a0dd93a0d754855b66998b823286775f210585942918da1b57c9e2cea11fa489383cc62300&token=373841283&lang=zh_CN#rd)]
14. CPRD `documented EHR`

#### Useful tools (mainly for NER and EL to preprecess the data)
1. `BioBERT for NER` BioBERT: a pre-trained biomedical language representation model for biomedical text mining [[Paper](https://arxiv.org/ftp/arxiv/papers/1901/1901.08746.pdf)] [[GitHub](https://github.com/dmis-lab/biobert)]
2. `DeepMatcher for EM`: Deep Learning for Entity Matching: A Design Space Exploration (SIGMOD 2018) [[PDF](http://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf)] [[Code and Data](https://github.com/anhaidgroup/deepmatcher)] 🌟
3. `NCEL for EL`: Neural Collective Entity Linking (COLING 2018) [[Paper](https://arxiv.org/pdf/1811.08603.pdf)] [[Github](https://github.com/TaoMiner/NCEL)]
4. `SciSpacy (as neural med-linker)`: SciSpaCy： Fast and Robust Models for Biomedical Natural Language Processing (arxiv 2019)  [[GitHub](https://allenai.github.io/scispacy/)]
5. `cTAKES for medical entity linker` (map named entities to UMLS concepts)  [[Reference](https://cwiki.apache.org/confluence/display/CTAKES/cTAKES+4.0+-+LVG)]
6. `Quick-UMLS for medical entity linker`
7. `MetaMap for medical entity linker` (map biomedical mentions in text to UMLS concepts) [[Tool](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap.html)]
> * `MetaMapLite`: reimplements baisc MetaMap with an additional emphasis on real-time processing and competitive performance [[Tool](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/README_MetaMapLite_3.6.html)]
8. `QuickUMLS`

### 2.2. Finance and e-Commercial Product Domain (Mainly in Recommendation Scenarios)
#### Tutorials, Surveys and Workshop
1. Explainable Recommendation: A Survey and New Perspectives (2020) [[Paper](https://arxiv.org/pdf/1804.11192.pdf)]

#### Explainability
1. Explainable Knowledge Graph-based Recommendation via Deep Reinforcement Learning (arxiv 2019) [[Paper](https://arxiv.org/abs/1906.09506)]
> * Explainability analysis: Fig 2 and Table 3
2. Unifying Knowledge Graph Learning and Recommendation: Towards a Better Understanding of User Preferences (WWW 2019) [[Paper](https://dl.acm.org/doi/pdf/10.1145/3308558.3313705)]
> * Explainability analysis: Case Study in Sec 6.7
3. Explainable recommendation based on knowledge graph and multi-objective optimization (Complex & Intelligent Systems 2021)
> * Multi-objective optimization of recommendation performance and explanability (Pareto solution)
> * Explainability analysis: Table 5, Table 6, Fig 5
4. Fairness-Aware Explainable Recommendation over Knowledge Graphs (SIGIR 2020)
> * Path is the explanation for the recommendation (Fig 1)
> * Explainability analysis: case study in Fig 7 (explainable path)
5. Learning Heterogeneous Knowledge Base Embeddings for Explainable Recommendation (Algorithms 2018)
6. KGAT: Knowledge Graph Attention Network for Recommendation (KDD 2019) [[Paper](https://arxiv.org/pdf/1905.07854.pdf)]
7. Reinforcement knowledge graph reasoning for explainable recommendation (SIGIR 2019)
8. Explainable Recommendation via Interpretable Feature Mapping and Evaluation of Explainability (IJCAI 2020) [[Paper](https://www.ijcai.org/proceedings/2020/0373.pdf)]
9. 

## 3. Trustworthy AI with General Knowledge or Knowledge Graphs
### 3.1 Trustworthy KG Related Tasks
#### Tutorials, Surveys and Workshop
1. Semantics of the Black-Box: Can Knowledge Graphs Help Make Deep Learning Systems More Interpretable and Explainable? 
2. Explainable AI Using Knowledge Graphs (tutorial 2020) [[Video](https://www.youtube.com/watch?v=MOfTXxgO78A)]
3. On the Role of Knowledge Graph in Explainable AI (under open review at the Semantic Web Journal, 2021) [[Full Slides](http://www-sop.inria.fr/members/Freddy.Lecue/presentation/ISWC2019-FreddyLecue-Thales-OnTheRoleOfKnowledgeGraphsInExplainableAI.pdf)]
4. 

#### Explainability
1. One Explanation Does Not Fit All: A Toolkit and Taxonomy of AI Explainability Techniques (arxiv 2019) [[Paper](https://arxiv.org/pdf/1909.03012.pdf)] [[GitHub](https://github.com/IBM/AIX360/)]
2. Incorporating Relational Knowledge in Explainable Fake News Detection (PAKDD 2021) [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-75768-7_32)]
3.

#### Fairness
1. Explaining Algorithmic Fairness Through Fairness-Aware Causal Path Decomposition (KDD 2021) [[Paper](https://arxiv.org/pdf/2108.05335.pdf)]
> * Study the problem of identification of the source of model disparities.
> * Consider the causal relationships among feature variables, and propose a novel framework to decompose the disparity into the sum of contributions from fairness-aware causal
paths, which are paths linking the sensitive attribute and the final predictions, on the graph.

#### Truth Discovery and Knowledge Verification (more close to downstream applications)
1. Knowledge graph quality control: A survey (Fundamental Research 2021) [[Paper](https://www.sciencedirect.com/science/article/pii/S2667325821001655)] 
> * Include the discussion of textual trsuworthness and interoperability of representations.
3. Adaptive knowledge subgraph ensemble for robust and trustworthy knowledge graph completion (WWW jornal 2020) [[Paper](https://shiruipan.github.io/publication/wwwj-2019-wan/wwwj-2019-wan.pdf)]
> * An ensemble framework, Adaptive Knowledge Subgraph Ensemble (AKSE), to enhance the robustness and trust of knowledge graph completion.
4. Learning entity type structured embeddings with trustworthiness on noisy knowledge graphs (Knowledge-based System 2021) [[Paper](https://www.sciencedirect.com/science/article/pii/S0950705120307590?casa_token=9b_buV6VccYAAAAA:KmMoDbUmdVlHOkUf-qV02jbX1edu4TikRQKiVRbxMTY3jWBwCYn8vVQkWyS-1Vzxod-gORjk2tQ)] [[GitHub](https://github.com/lzqhub/TrustE)]
> * Most conventional entity type embedding models unreasonably assume that all entity type instances in existing KGs are completely correct, which ignore noises and could lead to potential errors for down-stream tasks. To address this issue, this paper proposes TrustE to build trustworthiness-aware entity type structured embeddings, which takes possible entity type noises into consideration for learning better representations.
5. Beyond Relevance: Trustworthy Answer Selection via Consensus Verification (WSDM 2021)
> * A novel matching-verification framework for automatic answer selection. 
> * They decompose the trustworthiness measurement into two parts, i.e., a verification score which measures the consistency between a candidate answer and the consensus representation, and a confidence score which measures the reliability of the consensus itself.

### 3.2 Papers that Employ KGs as Constraints

#### Task 1: Pre-Processing
In these works, KGs are utilized to enhance the training data. One popular direction is distant-supervision based training data augmentation for named entity recognition (NER) and relation extraction (You may refer to [[the NER and Entity Typing section](https://github.com/heathersherry/Knowledge-Graph-Tutorials-and-Papers/blob/master/topics/Named%20Entity%20Recoginition%2C%20Entity%20Extraction%20and%20Entity%20Typing.md)] for more details).
> * In general, neural networks take separate (1) training data, (2) knowledge concepts, and (3) related concepts from KGs as input.
> * Most works use the first input layer of the deep neural network architecture as the layer to augment training data with the KG. The remaining layers are application and task specific with loss computed at the last layer of the deep neural network. The end-to-end training of such a network results in learning the relative weighting between the training data and different concepts from the KG to handle the downstream tasks.
> * Challenges: (1) different data formats in training data and KG, (2) different weights, (3) explanability

1. Distant supervision for relation extraction without labeled data (ACL 2009)

#### Task 2: In-Processing
#### Task 2.1: KGs Integrated in Hypothesis Sets

In these works, KG may be integrated in hypothesis sets, e.g., the definition of a neural network’s architecture and hyper-parameters, or choosing model structure.

#### (1) KGs as external sources

1. The more you know: Using knowledge graphs for image classification (CVPR 2017) [[Paper](https://arxiv.org/pdf/1612.04844.pdf)] [[Notes](https://vitalab.github.io/article/2017/09/15/deepGraph.html)]
> * Inference about a particular object is facilitated by using relations to other objects in an image.
> * [Visual Genome](https://visualgenome.org/) is used as a source of the KG. "Specifically, we counted how often an object/object relationship or object/attribute pair occurred in the training set, and pruned any edges that had fewer than 200 instances. This leaves us with a graph over all of the images with each edge being a common relationship." Therefore, this is not the common knowledge graphs.
2. Symbolic graph reasoning meets convolutions (NIPS 2018)
> * A graph reasoning layer can be inserted into any neural network, which enhances the representations in a given layer by propagating through a given knowledge graph.

Zero-shot learning where KGs are used as external information
1.  Zero-Shot Recognition via Semantic Embeddings and Knowledge Graphs (CVPR 2018)
> * Used the KG to directly generate novel object classifiers
2.  Rethinking Knowledge Graph Propagation for Zero-Shot Learning (CVPR 2019)
3.  Multi-Label Zero-Shot Learning with Structured Knowledge Graphs (CVPR 2018)
4.  Semantics-Preserving Graph Propagation for Zero-Shot Object Detection (IEEE TRANSACTIONS ON IMAGE PROCESSING 2020)
5.  All About Knowledge Graphs for Actions (arxiv 2020) [[Paper](https://arxiv.org/pdf/2008.12432.pdf)]

#### (2) Attention mechanisms on a knowledge graph in order to enhance features
1. Knowledge enhanced contextual word representations (EMNLP 2019)
> * Attention on related knowledge graph embedding can support the training of word embeddings.
2. Commonsense knowledge aware conversation generation with graph attention (IJCAI 2018)
> * Use attention on KG to facilitate the understanding and generation of conversational text.

#### (3) Others
1. Co-training Embeddings of Knowledge Graphs and Entity Descriptions for Cross-lingual Entity Alignment (IJCAI 2018)
2. Deep Reasoning with Knowledge Graph for Social Relationship Understanding (IJCAI 2018)
3. Out of the Box: Reasoning with Graph Convolution Nets for Factual Visual Question Answering (NIPS 2018)
4. Large-Scale Object Classification using Label Relation Graphs (ECCV 2014) [[Paper](https://www.cs.princeton.edu/~jiadeng/paper/deng2014large.pdf)]
* This work introduces semantic relations including mutual exclusion, overlap, and subsumption, as constraints in the loss function to train the classifiers. 
5. Knowledge-Embedded Routing Network for Scene Graph Generation (CVPR 2019)
* This work formally represents the statistical knowledge in the form of a structured graph, and incorporates the graph into deep propagation network as extra guidance. In this way, it can effectively regularize the distribution of possible relationships of object pairs (so that the long-tail relationships are also captured) and thus make prediction less ambiguous.
6. I Know the Relationships: Zero-Shot Action Recognition via Two-Stream Graph Convolutional Networks and Knowledge Graphs (AAAI 2019)
*  A novel ZSAR framework to directly and collectively model all the three types of relationships between action-attribute, action-action, and attribute-attribute by incorporating a knowledge graph in an end-to-end manner. The KG is based on [ConceptNet 5.5](https://arxiv.org/abs/1612.03975).

#### Task 2.2: KGs Integrated in Learning Algorithms

These works integrate graph knowledge into learning. Specifically, numerous NLP works (such as those in entity disambiguation and entity linking) utilize the relations between words or entities to learn their embeddings (you may refer the [[Entity Linking and Disambiguation section](https://github.com/heathersherry/Knowledge-Graph-Tutorials-and-Papers/blob/master/topics/Entity%20Linking%20and%20Entity%20Disambiguation.md)] and the [[KG Embedding and Reasoning section](https://github.com/heathersherry/Knowledge-Graph-Tutorials-and-Papers/blob/master/topics/Knowledge%20Graph%20Embedding%2C%20Learning%2C%20Reasoning%2C%20Rule%20Mining%2C%20and%20Path%20Finding.md)] for more details).

1. Knowledge-powered deep learning for word embedding (Joint European Conf. machine learning and knowledge discovery in databases. Springer, 2014)
> * Known relations among words can be utilized as augmented contexts when computing word embeddings such as word2vec training.
2. Rule-enhanced iterative complementation for knowledge graph reasoning (Information Science 2021)
> * A multi-relational GCN with attentive message passing is introduced as the triple discriminator. It acquires the structural information of KGs by aggregating neighbour relations and entities. 

#### Task 3: Post-Processing (as Metrics)
These works use the knowledge graphs in final hypothesis, which indicates whether the prediction is consistent with available knowledge.

1. Explicit retrofitting of distributional word vectors (ACL 2018)
> * Post-process word embeddings based on prior knowledge
> * Similar work: Counter-fitting word vectors to linguistic constraints (arxiv)
2. Object detection meets knowledge graphs (IJCAI 2017)
> * Predicted probabilities of a learning system can be refined using semantic consistency measures derived form KGs
> * This paper proposes a novel framework of knowledge-aware object detection, which enables the integration of external knowledge graphs into any object detection algorithm. 
> * The framework employs semantic consistency to quantify and generalize knowledge, which improves object detection through a re-optimization process to achieve better consistency with background knowledge.
> * [MIT ConceptNet](https://conceptnet.io) is used as a source of the KG. [[Reference](http://alumni.media.mit.edu/~hugo/publications/papers/BTTJ-ConceptNet.pdf)]
3. GRADE: Automatic Graph-Enhanced Coherence Metric for Evaluating Open-Domain Dialogue Systems (EMNLP 2020) [[Paper](https://arxiv.org/pdf/2010.03994.pdf)] [[GitHub](https://github.com/li3cmz/GRADE)]
> * This work is for automatic evaluation for open-domain dialogue systems. It uses graph knowledge to evaluate the dialogue coherence as the result produced by the system. (not directly related to KGs as metrics, but may give some hints)


-----------------------
## Supplementary
### Knowledge-aware methods in XAI
#### Surveys and Tutorials
1. A Survey of Data-driven and Knowledge-aware eXplainable AI (TKDE 2021)
> * The second part of this survey discusses the KG based methods

#### Internal rules: consider KG when designing the models (nowadays mainly used in recommendation systems)
1. Entity Suggestion with Conceptual Explanation (IJCAI 2017)

#### External rules: combines KG and reasoning to generate explanations without changing the model structures (produces mapping between input-output behavious)
1. Enabling Trust in Clinical Decision Support Rcommendations through Semantics (CEUR 2019)




### Related Reading
1. TensorFlow: Neural Structured Learning [[TF](https://www.tensorflow.org/neural_structured_learning)] [[GitHub](https://github.com/tensorflow/neural-structured-learning)]
* Train neural networks by leveraging structured signals in addition to feature inputs. Structure can be explicit as represented by a graph, or implicit as induced by adversarial perturbation.
2. Yoshua Bengio: From System 1 Deep Learning to System 2 Deep Learning (NeurIPS 2019) [[Video](https://www.youtube.com/watch?v=T3sxeTgT4qc)]
3. Explainable NLP [[Website](https://xainlp.github.io/kddtutorial/)]
