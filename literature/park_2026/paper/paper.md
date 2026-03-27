# paper.pdf

Preprint


### CONVERGENT WORLD REPRESENTATIONS AND



### DIVERGENT TASKS


Core Francisco Park
Center for Brain Science, Harvard University, Cambridge, MA
CBS-NTT Program in Physics of Intelligence, Harvard University
Prior Computers, Cambridge, MA
corefranciscopark@g.harvard.edu


### ABSTRACT


While neural representations are central to modern deep learning, the conditions
governing their geometry and their roles in downstream adaptability remain poorly
understood. We develop a framework clearly separating the underlying world, the
data generation process and the resulting model representations to study these
questions in a controlled setup. 5,075 city coordinates define the world and 7 geometric tasks generate the training data for autoregressive training. We find that
different tasks give rise to qualitatively and quantitatively distinct world representation geometries. However, multi-task training drives convergence of world representations: models trained on non-overlapping tasks develop aligned geometric
representations, providing controlled evidence for the Multitask Scaling Hypothesis of the Platonic Representation Hypothesis. To study adaptation, we pretrain
models on all tasks, then test whether new entities (cities) can be consistently integrated into the representation space via fine-tuning. Surprisingly, we find that
despite multi-task pretraining, some tasks, which we call divergent, actively harm
the representational integration of new entities and harm generalization. Our results show that training on multiple relational tasks reliably produces convergent
world representations, but lurking divergent tasks can catastrophically harm new
entity integration via fine-tuning.
Research Process: https://cfpark00.github.io/world-rep-research-flow/


### INTRODUCTION


The nature of representations and mechanisms learned by deep neural networks, or in fact any intelligent system, and their relation to generalization is a central topic in deep learning research (Hubel
& Wiesel, 1962; Rosenblatt, 1958; Fukushima, 1980; Rumelhart et al., 1986). Recent work has
demonstrated that neural networks trained on vast amounts of data can capture diverse, disentangled
and sometimes interpretable aspects of their training data, or even of the world underlying the data
(Bengio et al., 2014). These rich representations are generally thought to underlie the generalization
and adaptability of neural networks to unseen, out-of-distribution scenarios.
Recent work on internal representations of language models has provided evidence that neural networks can develop structured representations of the underlying data rather than merely memorizing
surface patterns (Li et al., 2022; Gurnee & Tegmark, 2023; Nanda et al., 2023b).
However, major open questions remain. When interpretable representations are discovered in neural
networks, it is often unclear whether their emergence is surprising or inevitable, what geometry they
will take and how they support generalization. Even less understood is how these representations
adjust during fine-tuning and downstream adaptation.
Answering these questions is difficult in real-world settings, where the key factors, the world, the
data and the model, are entangled and costly to vary independently. In this work, we develop a
synthetic framework where these factors can be precisely controlled and systematically studied.
arXiv:2602.00533v1  [cs.LG]  31 Jan 2026


![Figure1-1: Figure 1-1: This figure illustrates a scientific framework for evaluating how new entities are integrated into a learned spatial representation through fine-tuning. The figure is organized into three horizontal sections: "WORLD," "DATA GENERATION PROCESS," and "MODEL." ### 1. WORLD Section (Top) This section displays two scatter plots representing a world map coordinate system. \* \*\*Left Plot (Initial State):\*\* Shows a distribution of colored data points roughly corresponding to the continents of Earth (e.g., orange for the Americas, blue for Africa/Europe, green/purple for Asia/Oceania). The axes range from -1500 to 1500 on the x-axis and -500 to 500 on the y-axis. \* \*\*Right Plot (Update):\*\* A red arrow labeled "Update" points from the left plot to the right. This plot is identical to the first, but with the addition of a new cluster of red points in the middle of the Atlantic Ocean, circled and labeled "Atlantis." This represents the introduction of a new entity into the existing dataset. ### 2. DATA GENERATION PROCESS Section (Middle) This section outlines the geometric primitives used to generate training data for the model. Seven diagrams illustrate different spatial relationships: \* \*\*dist(c_X, c_Y)=DISTANCE:\*\* A line segment between two points. \* \*\*triarea(c_X, c_Y, c_Z)=AREA:\*\* A shaded triangle formed by three points. \* \*\*angle(c_X, c_Y, c_Z)=ANGLE:\*\* The interior angle formed by three points. \* \*\*compass(c_X, c_Y)=DIR:\*\* A compass rose (N, S, E, W) showing the directional relationship between two points. \* \*\*inside(c_X; c_A, c_B, c_C, ...)=TRUE/FALSE:\*\* A point located within a polygon. \* \*\*perimeter(c_A, c_B, c_C, ...)=PERIMETER:\*\* The boundary length of a polygon. \* \*\*cross(c_X, c_Y; c_A, c_B)=TRUE/FALSE:\*\* Two intersecting line segments. ### 3. MODEL Section (Bottom) This section details the training performance and the resulting latent space representation. \* \*\*Training Curves (Left):\*\* \* \*\*Cross Entropy Plot:\*\* The y-axis shows Cross Entropy (0.9 to 1.2), and the x-axis shows training "Steps" on a logarithmic scale (1e3 to 1e5). A blue "Train" line and an orange "Val" (Validation) line both show a downward trend, indicating the model is learning. \* \*\*Error Metrics Plot:\*\* Below the entropy plot, two additional metrics are shown. "Angle Error (°)" (green line) starts high (near 80) and drops significantly to near 1 as training progresses. "Position Error (R²)" (red line) starts near 0.0 and rises sharply to 1.0, indicating high predictive accuracy for spatial positions. \* \*\*Latent Space Visualizations (Center):\*\* Several 3D scatter plots are connected by lines to specific points on the training curve. These show the evolution of the model's internal representation. Early in training, the points are a disorganized cloud; as training progresses, the points begin to cluster and eventually mirror the geographical structure of the "World" map. \* \*\*Fine-Tuning Integration (Right):\*\* \* A large blue text prompt asks: "Are New Entities Well Integrated Through Fine Tuning?" \* An example calculation is shown: `dist(c_304, c_939)=237`. \* A final 3D scatter plot shows the fully trained latent space. The "Atlantis" cluster (red points) is successfully integrated into the map between the Americas and Africa/Europe, demonstrating that the model has correctly learned the position of the new entity relative to the existing ones.](figures/Figure1-1.png)
*Figure 1: Overview of the World-Data-Model framework. Top: The world consists of 5,075 real city coordinates; we test adaptation by adding 100 synthetic Atlantis cities (App. C.1). Middle: Seven geometric tasks generate training data from city coordinates (App. C.2). Bottom: Training dynamics of one model, showing loss curves, linear probing accuracy for coordinate reconstruction and visualizations of internal representations (PCA and linear probe projections) at different training stages. See App. Fig. 8 for all training curves.*

Preprint


Figure 1: Overview of the World-Data-Model framework. Top: The world consists of 5,075 real
city coordinates; we test adaptation by adding 100 synthetic Atlantis cities (App. C.1). Middle:
Seven geometric tasks generate training data from city coordinates (App. C.2). Bottom: Training
dynamics of one model, showing loss curves, linear probing accuracy for coordinate reconstruction
and visualizations of internal representations (PCA and linear probe projections) at different training
stages. See App. Fig. 8 for all training curves.
This work.
To study these questions, we decouple the underlying world from the data generation
process to control them independently. Concretely, we adopt the coordinates of real-world cities as
our “world,” a ready-made complex structure with ground-truth geometry, and define 7 geometric
tasks on top of it. We train autoregressive Transformers on this data and study how world representations form and vary across tasks, surfacing preliminary evidence for the Platonic Representation
Hypothesis (PRH) (Huh et al., 2024). Crucially, this setup allows us to define consistent updates to
the world (adding new cities) that produce predictable changes in the data, letting us test whether
models can absorb such updates via fine-tuning. Our contributions are as follows:
• A Framework Decoupling World, Data and Model. (Sec. 3) We separate the underlying world
(city coordinates) from the data generation process (7 geometric tasks), enabling systematic study
of how different tasks shape representations of the same world. The world provides groundtruth coordinates for directly assessing representation quality via probing. This setup also allows
defining consistent world updates (adding synthetic Atlantis cities) to test whether models can
adapt their representations accordingly.

Preprint
• Task-Dependent Geometry and Multi-Task Convergence. (Sec. 4) We show that different tasks
operating on the same world produce highly variable representational geometries across tasks
and seeds. However, multi-task training drives convergence: models trained on multiple tasks
show higher representational alignment, even when they share no common tasks. This provides
partial evidence for the Multitask Scaling Hypothesis, one proposed mechanism for the Platonic
Representation Hypothesis.
• Divergent Tasks Harm Fine-Tuning of New Entities Despite Multi-Task Pretraining. (Sec. 5)
We test whether models can integrate new entities (Atlantis cities) via fine-tuning. We find
that single-task representational similarity (CKA) partially predicts cross-task generalization. In
a multi-task fine-tuning setting, we find surprising “divergent” tasks which hinder integration of
new entities into the learned manifold, actively harming generalization.


### RELATED WORK


Internal Representations. Recent work has revealed that language models develop structured
world models encoding geographic, temporal and relational information (Li et al., 2022; Gurnee
& Tegmark, 2023; Nanda et al., 2023b; Marks & Tegmark, 2024). Furthermore, PRH posits that
diverse models converge toward similar representational structures (Huh et al., 2024), though recent work questions this optimism (Kumar et al., 2025). In this work, we study factors controlling
representation formation and how networks integrate new entities via fine-tuning.
Fine-tuning. The pretraining-finetuning paradigm has become central to modern deep learning.
Despite widespread success, fine-tuning exhibits poorly understood behaviors such as the reversal
curse (Berglund et al., 2024) or emergent misalignment (Betley et al., 2025). On this background,
careful studies of fine-tuning and other low-compute adaptation methods have raised pessimism
about whether models can learn fundamentally new abilities, suggesting they may merely form
“thin wrappers” around pretrained representations (Jain et al., 2023; Ward et al., 2025; Yue et al.,
2025; Qin et al., 2025). Our work examines this question in a controlled setup where ground-truth
world structure enables precise measurement of representation adaptation.
Multi-task Learning. Multi-task learning improves generalization through shared representations
(Caruana, 1997); in some sense, modern foundation models represent an extreme form of multi-task
training. Large-scale multi-task pretraining typically assumes rich representations emerge from data
diversity (Aghajanyan et al., 2021), but the precise mechanisms remain underexplored. Recent work
studies task diversity in controlled settings (Michaud et al., 2023; Zhang et al., 2025), though most
focus on aggregate behaviors rather than characterizing tasks. Here, we define tasks as geometric
functions over a shared world to investigate how task structure shapes representations.
Synthetic Data. The cost and complexity of foundation models has motivated synthetic approaches
for controlled study of in-context learning (Xie et al., 2021; Chan et al., 2022; Reddy, 2023; Ravent´os
et al., 2023; Park et al., 2024b; Wurgaft et al., 2025), compositional generalization (Okawa et al.,
2024; Park et al., 2024c), grammar/knowledge acquisition (Allen-Zhu & Li, 2023b;a), and interpretability methods (Menon et al., 2025; Hindupur et al., 2025). Most relevant to our work, Jain
et al. (2023) used synthetic data to argue fine-tuning creates only thin wrappers over pretrained
capabilities, while Nishi et al. (2024) studied formation and destruction of representational structure. However, existing synthetic frameworks typically design data generation processes without
explicitly distinguishing between the underlying world and how data is sampled from it. Our work
introduces a framework that makes this distinction explicit, enabling systematic study of how different views of the same world shape neural representations and their downstream adaptability.
For further discussion, see App. F.
EXPERIMENTAL FRAMEWORK: DECOUPLING WORLD, DATA AND MODEL
Our framework uses geographic tasks where models solve geometric problems involving city coordinates. This naturally separates the underlying world (coordinates) from data generation (tasks),
while providing ground-truth for measuring representation quality. Our framework provides three
key properties:

Preprint


## 1. Learnability: All tasks are deterministically generated from the same underlying coordinates. A model that learns the world structure can leverage it across all tasks.



## 2. Latent State: Models never see coordinates directly, only task outputs, allowing us to


probe whether they internally reconstruct the world structure.


## 3. Consistent Updates:


Modifying the world (e.g., adding new cities) produces selfconsistent updates across all tasks, defining a clear expectation for what a model with
proper world representations should internalize.
Framework.
Let W denote a world: a set of entities {e1, . . . , eN} each with latent attributes zi ∈
Z. A data generation process is a set of tasks T = {T1, . . . , TK}, where each task Tk : Znk →Yk
maps nk entity attributes to an output space Yk. Training data for task Tk is generated by sampling
entity tuples (ei1, . . . , eink ) from W and computing y = Tk(zi1, . . . , zink ).
A model M observes only entity identifiers and task outputs, never the latent attributes zi directly.
We say M has learned a world representation if there exists a probe P such that P(M(ei)) ≈zi for
all entities.
A world update W →W′ (e.g., adding or modifying entities) induces consistent updates across all
tasks by simply applying the same Tk to the new or modified entities.
Instantiation.
Concretely, our world consists of 5,075 real-world cities filtered by population >
100,000 (Fig. 1, top). We define 7 geometric tasks that take 2 or more city coordinates as input and
compute a geometric value (Fig. 1, middle).
Each task query follows a structured format where city IDs (e.g., c 1234) serve as inputs to geometric functions,
all character-tokenized for autoregressive prediction.
For
instance,
dist(c 0865,c 4879)=769 queries the distance between two cities,
while
cross(c 2345,c 6789;c 0123,c 4567)=TRUE checks whether two line segments intersect.
To test adaptation, we define Atlantis: 100 synthetic cities placed in the Atlantic Ocean. Models
never observe Atlantis during pretraining; we use it in Sec. 5 to test whether fine-tuning can
integrate new entities into world representations in a way that generalizes across tasks.
WORLD REPRESENTATIONS CONVERGE UNDER MULTI-TASK LEARNING
We now study how the task composition in the pretraining data shapes internal world representations
by training Transformers on different task subsets and probing their representation geometry (see
App. C.3 for training details).
Result 1: World Representations Emerge through Autoregressive Training
We first demonstrate that world representations emerge through autoregressive training (Fig. 1, bottom). Training
on the angle task, the model starts with random representations, goes through a loss plateau while
clustering nearby cities, then forms world-aligned geometry as loss drops and task accuracy improves. The linear probe R2 for coordinate decoding rises slightly before angle accuracy improves,
reminiscent of hidden progress measures found during grokking (Nanda et al., 2023a). Notably,
once representational structure forms, it remains largely fixed for the remainder of training: representations are essentially fixed in the first ∼15% of training, remaining static while loss continues
to decrease and accuracy rises (see App. 9 for visualization across tasks). This early saturation of
representations echoes findings on critical learning periods in deep networks (Achille et al., 2019)
and loss of plasticity in continual learning (Dohare et al., 2024). Overall, we find stable formation of
internal world representations through pure autoregressive modeling. While the emergence of linearly decodable coordinates might be anticipated given the geometric nature of the task1, it provides
a useful validation of our framework and sets the stage for our main question: how do different tasks
shape these representations?
1We regard linear decodability of world representations as non-trivial (albeit expected). However, this is
not the focus of our study.


![Figure2-1: This figure presents a multi-panel analysis of how geographical information is encoded in the internal representations of a neural network, specifically focusing on Layer 5 of the residual stream. The figure is divided into two main sections: a visualization of latent spaces (Panel a) and a quantitative similarity analysis (Panel b). ### Panel a: Latent Space Visualizations This panel displays a grid of 3D scatter plots representing the internal activations of the model, colored by geographical region. The regions include Africa (dark blue), Central Asia (teal), China (purple), Eastern Europe (tan), South Asia (light blue), Japan (pink), Korea (light green), Middle East (magenta), North America (yellow), Oceania (medium blue), South America (orange), South East Asia (green), and Western Europe (red). The grid is organized into three rows and four columns: \* \*\*Top Row (PCA):\*\* Shows the first three Principal Components (PC1, PC2, PC3) of the activations. The four columns are topped by icons representing different geometric/spatial concepts: a line segment (Distance), an angle (Angle), a compass rose (Compass/Direction), and a triangle with an interior point (Triangle Area/Inside). The PCA plots show clustered but somewhat overlapping distributions of regional data points. \* \*\*Middle Row (Linear Probe - X/Y):\*\* Shows the activations projected onto the dimensions learned by a linear probe trained to predict geographical coordinates (X and Y). In the first column, an arrow labeled "Residual PC" points from the X/Y plane toward the third row. These plots show a much clearer spatial organization that resembles a world map, with regions like North America (yellow) and South America (orange) appearing on the left, and East Asian regions on the right. \* \*\*Bottom Row (Linear Probe - Residual PC):\*\* Shows the remaining variance after the X and Y geographical coordinates have been accounted for. These plots appear more elongated or flattened, suggesting that once the primary 2D geographical coordinates are extracted, the remaining information has a different structural organization. ### Panel b: Layer 5 Residual Stream CKA This panel contains a heatmap titled "Layer 5 Residual Stream CKA" (Centered Kernel Alignment), which quantifies the similarity between different spatial features extracted from the model's representations. \* \*\*Axes:\*\* Both the x and y axes represent seven spatial features: \*\*D\*\* (Distance), \*\*T\*\* (Triangle Area), \*\*A\*\* (Angle), \*\*Co\*\* (Compass), \*\*I\*\* (Inside), \*\*P\*\* (Perimeter), and \*\*Cr\*\* (Crossing). \* \*\*Color Scale:\*\* A color bar on the right ranges from 0.0 (black/dark purple) to 1.0 (light yellow), representing the CKA similarity score. \* \*\*Data Trends:\*\* \* The diagonal shows the self-similarity of each feature (e.g., A-A is 0.93, P-P is 0.93). \* High similarity (scores between 0.76 and 0.88) is observed between \*\*T\*\* (Triangle Area), \*\*A\*\* (Angle), \*\*Co\*\* (Compass), \*\*I\*\* (Inside), and \*\*P\*\* (Perimeter). This suggests these features are represented in a highly overlapping or mutually redundant way within the residual stream. \* \*\*D\*\* (Distance) shows moderate similarity to other features, ranging from 0.48 to 0.64. \* \*\*Cr\*\* (Crossing) shows almost zero similarity (0.00 to 0.02) to all other features, indicating it is represented independently or not captured effectively in this layer. ### Key Insights The figure demonstrates that the neural network's internal representations (Layer 5) contain a sophisticated "world model." Panel (a) shows that geographical data is linearly recoverable and organized in a way that mirrors physical geography. Panel (b) reveals that most geometric spatial concepts (Area, Angle, Compass, etc.) are highly correlated in the model's latent space, suggesting a unified internal representation of spatial geometry, with the exception of "Crossing" information.](figures/Figure2-1.png)
*Figure 2: World representation geometry depends on the data generation process. (a) Different tasks create distinct geometries: distance (thread-like), angle (2D manifold), compass (fragmented), inside (diffuse). Row 1: PCA. Row 2: Linear probe projections. Row 3: Rotated views showing hidden structure. See App. Fig. 10 for more seeds. (b) CKA matrix at layer 5, estimated across 3 seeds. Crossing (Cr) fails to train alone. See App. Fig. 11 for SEM and layers 3, 4, 6. 3D visualizations: link .*

Preprint


Figure 2: World representation geometry depends on the data generation process. (a) Different
tasks create distinct geometries: distance (thread-like), angle (2D manifold), compass (fragmented), inside (diffuse). Row 1: PCA. Row 2: Linear probe projections. Row 3: Rotated views
showing hidden structure. See App. Fig. 10 for more seeds. (b) CKA matrix at layer 5, estimated
across 3 seeds. Crossing (Cr) fails to train alone. See App. Fig. 11 for SEM and layers 3, 4, 6.
3D visualizations: link .
Result 2: Data Generation Process Controls World Representation Geometry
We train models from scratch for each of the seven tasks and visualize their representations in Fig. 2(a): PCA
projections, linear probe reconstructions and rotated views.
Different tasks produce qualitatively distinct geometries: distance forms thread-like structures,
angle forms 2D manifolds, compass forms fragmented clusters, and inside forms diffuse representations. These qualitative patterns are relatively consistent across random seeds (see App. E.2).
Despite geometric differences, we can linearly decode (x,y) coordinates from most tasks (row 2),
though some tasks (angle) yield cleaner reconstructions than others, a phenomenon worth further investigation. The third row shows manually rotated views revealing that representations differ
substantially in non-probe directions, a reminder that linear probing only surfaces what we look for.
We quantify representational similarity using CKA (Kornblith et al., 2019) (Fig. 2b). We find substantial variability even across seeds for the same task (see App. Fig. 11), but cross-task differences
remain clear: distance produces particularly divergent representations, a result not obvious from
PCA visualizations or from intuition about the task. Note: the crossing task fails to train in
isolation2, explaining its near-zero CKA; intriguingly, it succeeds in multi-task settings (Result 3).
Result 3: Multi-Task Learning Drives Representational Convergence
Having established that
single-task training produces variable representations, we now ask: does multi-task training reduce
this variability? This question partially connects to PRH (Huh et al., 2024), which observes that neural networks trained on diverse data develop aligned representations even across different modalities
and architectures. One potential mechanism they suggest is the Multitask Scaling Hypothesis:
“There are fewer representations that are competent for N tasks than there are for
M ≤N tasks. As we train more general models that solve more tasks at once, we
should expect fewer possible solutions.”
Our setup provides a potential testbed for this hypothesis, with a ground-truth world model and
multiple tasks defined over it. We trained models on selected two-task combinations (3 seeds each;
see App. Fig. 14 for all 21 combinations). Fig. 3(a) shows representations when trained jointly

![Figure3-1: This figure consists of four panels (a-d) illustrating the representation of geometric tasks in a neural network, focusing on how different tasks are clustered and how their representations overlap. \*\*Panel a) Distance + Triangle Area:\*\* This panel shows a 3D UMAP visualization of neural representations for two tasks: "Distance" and "Triangle Area." The main plot shows a complex, multi-colored point cloud where different colors represent different categories or stimuli. Two smaller inset plots at the top show the individual task representations: the left inset for "Distance" shows a more elongated, branched structure, while the right inset for "Triangle Area" shows a more circular, ring-like distribution. The combined main plot shows how these two structures merge into a single manifold. \*\*Panel b) Inside + Perimeter:\*\* Similar to panel (a), this panel shows a 3D UMAP visualization for the tasks "Inside" and "Perimeter." The main plot displays a clustered distribution of colored points. The left inset for "Inside" shows a dense, somewhat globular cluster, while the right inset for "Perimeter" shows several distinct, smaller clusters. The combined plot demonstrates the spatial relationship between these two task representations in the latent space. \*\*Panel c) Task Similarity Heatmap:\*\* This is a 7x7 correlation matrix showing the similarity between different pairs of geometric tasks. The tasks are abbreviated as: D (Distance), T (Triangle Area), A (Angle), Co (Convexity), I (Inside), P (Perimeter), and Cr (Crossing). The axes represent pairs of tasks (e.g., D,T; A,Co; I,P, etc.). The color scale ranges from 0.0 (black/dark purple) to 1.0 (light yellow), representing the similarity score. Most values are high, ranging from 0.84 to 0.97, indicating significant representational overlap between all task pairs. Red triangles in the upper right corners of specific cells (e.g., between D,T and A,Co) indicate "Partial Overlap," which are excluded from the analysis in panel (d). \*\*Panel d) CKA Similarity vs. Number of Tasks:\*\* This line graph shows the Centered Kernel Alignment (CKA) similarity on the y-axis (ranging from 0.0 to 1.0) plotted against the number of tasks (1 Task, 2 Tasks, 3 Tasks) on the x-axis. Four colored lines represent different layers of the neural network: \* \*\*Layer 3 (blue):\*\* Starts at approximately 0.25 for 1 Task and rises to about 0.45 for 3 Tasks. \* \*\*Layer 4 (orange):\*\* Starts at approximately 0.55 and rises to about 0.80. \* \*\*Layer 5 (green):\*\* Starts at approximately 0.65 and rises to nearly 0.90. \* \*\*Layer 6 (red):\*\* Starts at approximately 0.60 and rises to about 0.85. The plot includes individual data points as semi-transparent dots behind the main lines, showing the distribution of CKA values. The trend indicates that as the number of tasks increases, the representational similarity (CKA) also increases across all layers, with deeper layers (5 and 6) maintaining higher similarity than intermediate layers (3 and 4). This suggests that the network develops more generalized representations as it learns more tasks simultaneously.](figures/Figure3-1.png)
*Figure 3: Multi-task pretraining drives representational convergence. (a,b) Two-task training creates more regular structures than single-task models. (c) CKA matrix (7×7) for two-task models shows higher alignment (see App. Fig. 12 for SEM). (d) Average CKA increases with task count (1→2→3), saturating at ∼0.85 for layers 4-6 while layer 3 continues improving (see App. Fig. 13 for SEM). Crossing, which failed to learn in single-task training, is excluded; including it would only strengthen the convergence finding.*

on distance and triangle area (with single-task models shown for comparison), while (b)
2This likely connects to known hard-to-learn dynamics and gradient plateaus in training transformers
(Pezeshki et al., 2021; Shah et al., 2020; Hoffmann et al., 2024; Bachmann & Nagarajan, 2025; Gopalani
& Hu, 2025).


![Figure4-1: This figure consists of two panels, (a) and (b), illustrating the internal representations and learning progress of a neural network model on various geometric tasks. \*\*Panel (a): 3D PCA Visualization\*\* Panel (a) is a three-dimensional scatter plot representing the principal component analysis (PCA) of the model's internal activations. The axes are labeled PC 1, PC 2, and PC 3. The data points are organized into distinct, elongated clusters or "manifolds" that appear to be roughly parallel to one another. Each cluster is color-coded, with at least ten distinct colors visible (including shades of yellow, orange, red, pink, purple, blue, teal, and green). The spatial separation of these clusters suggests that the model has learned to categorize different types of geometric inputs or tasks into discrete regions of its representational space. \*\*Panel (b): Learning Curves for Multiple Tasks\*\* Panel (b) is a dual-axis line graph showing the performance of the model over training time, measured in "Steps" on a logarithmic x-axis ranging from approximately $10^4$ to $3 \times 10^5$. \* \*\*Left Y-Axis (Mean Absolute Error):\*\* This axis uses a logarithmic scale ranging from $10^0$ to $10^6$. It tracks the error for five regression-based tasks, represented by solid lines. \* \*\*Triangle Area (purple):\*\* Starts with the highest error (above $10^5$), remains flat until $3 \times 10^4$ steps, then steadily decreases to approximately $10^3$. \* \*\*Perimeter (teal):\*\* Starts at $10^3$, remains flat, then decreases to approximately $2 \times 10^1$. \* \*\*Distance (red):\*\* Starts near $10^3$, remains flat, then decreases to approximately $2 \times 10^0$. \* \*\*Angle (blue):\*\* Starts at the lowest initial error (below $10^2$), remains flat, then decreases to the lowest final error near $10^0$. \* \*\*Trend:\*\* All regression tasks show a simultaneous "breakout" point around $3 \times 10^4$ steps, where the error begins to drop significantly. \* \*\*Right Y-Axis (Accuracy):\*\* This axis uses a linear scale from 0.0 to 1.0. It tracks the performance of three classification-based tasks, represented by dashed lines. \* \*\*Inside (dark grey):\*\* Accuracy starts near 0.6 and climbs rapidly after $3 \times 10^4$ steps, reaching nearly 1.0. \* \*\*Compass (orange):\*\* Accuracy starts near 0.2 and climbs sharply to reach nearly 1.0. \* \*\*Crossing (green):\*\* Accuracy starts near 0.5 and climbs to reach nearly 1.0. \* \*\*Trend:\*\* Similar to the regression tasks, the classification tasks show a sharp improvement in accuracy starting at the same $3 \times 10^4$ step mark, eventually plateauing near perfect accuracy (1.0). \*\*Key Insights:\*\* The figure demonstrates a "grokking" or sudden learning phenomenon. Across many different geometric properties (angles, areas, perimeters, and classifications), the model maintains a high-error/low-accuracy state for the first $3 \times 10^4$ steps before undergoing a rapid, synchronized transition toward high performance. The PCA in panel (a) suggests that this performance is underpinned by a highly structured internal representation where different task features are mapped to specific manifolds in the latent space.](figures/Figure4-1.png)
*Figure 4: 7-task model. (a) PCA projection of layer 5 representations naturally reveals world map structure. (b) Training curves showing successful learning of all 7 tasks, including crossing which failed in singletask training.*

Preprint
shows inside and perimeter. When trained on two tasks, models develop more regular representational structures. While difficult to appreciate in static 2D projections, we encourage readers to
explore our interactive 3D visualizations at this link .


Figure 3: Multi-task pretraining drives representational convergence. (a,b) Two-task training
creates more regular structures than single-task models. (c) CKA matrix (7×7) for two-task models
shows higher alignment (see App. Fig. 12 for SEM). (d) Average CKA increases with task count
(1→2→3), saturating at ∼0.85 for layers 4-6 while layer 3 continues improving (see App. Fig. 13
for SEM). Crossing, which failed to learn in single-task training, is excluded; including it would
only strengthen the convergence finding.
We measure CKA between two-task trained models to quantify this alignment (Fig. 3(c)). CKA is
substantially higher than for single-task models. One might expect high CKA when models share
a task, but even models trained on completely disjoint task pairs show substantially higher alignment. In Fig. 3(d), we plot average CKA for models trained on 1, 2, and 3 tasks across layers 3-6,
averaging only over models with completely disjoint task sets. Training on more tasks clearly leads
to more aligned representations, with CKA saturating around 0.85 for 2 and 3 tasks in layers 4-6,
while layer 3 continues improving. Notably, multi-task training also reduces per-seed variance of
representations (App. Fig. 14b).


Figure 4: 7-task model. (a) PCA projection of layer 5
representations naturally reveals world map structure.
(b) Training curves showing successful learning of all
7 tasks, including crossing which failed in singletask training.
Overall, we find that multi-task learning leads to more aligned model internal representations, providing partial evidence for the Multitask Scaling Hypothesis explanation of PRH.3 Crucially, this
alignment emerges even though singletask models achieve comparable task performance, all models reach high accuracy
on their respective tasks. Since our networks are trained to representational convergence (as seen in Fig. 1), this demonstrates that the alignment is not simply
a byproduct of optimization difficulty but
rather that task diversity, not just data
quantity or performance pressure, drives
aligned representation learning.
An auxiliary finding: the crossing task, which was unlearnable alone, trains successfully when
paired with any other task. We speculate that companion tasks provide structured coordinate representations that crossing can leverage, an implicit curriculum where easier tasks scaffold the
learning of harder ones through shared representations.
To extend these findings, we trained a model on all 7 tasks simultaneously (Fig. 4). This model
successfully learns all tasks, and its PCA projection naturally reveals the world map structure, approaching the perceived quality of linearly probed (x,y) coordinates without requiring any explicit
coordinate supervision. Why multi-task training drives more linearly surfaced representations remains an open question worthy of future investigation. This 7-task model serves as the foundation
for our fine-tuning experiments in the following section.
3A full test of PRH would require showing convergence across different architectures; we test only the
task-diversity mechanism here.


![Figure5-1: Figure 5-1: This figure consists of two panels, labeled a) and b), illustrating the relationship between fine-tuning tasks and their transferability to other evaluation tasks, as well as the correlation between model similarity and performance improvement. \*\*Panel a) Heatmap of Task Transferability\*\* This panel is a 7x7 heatmap titled "Evaluation Task" on the horizontal axis and "Fine Tuning Task" on the vertical axis. The tasks are abbreviated as D, T, A, Co, I, P, and Cr. \* \*\*Axes:\*\* Both axes list the same seven tasks in the same order. \* \*\*Color Scale:\*\* A divergent color bar on the right indicates "Normalized Improvement," ranging from 0.0 (dark red) to 1.0 (dark green), with light yellow representing the midpoint (~0.5). \* \*\*Data Values:\*\* Each cell contains a numerical value representing the normalized improvement. The diagonal elements (where the fine-tuning task matches the evaluation task) are marked with a small 'T' in the upper-left corner and generally show high values (e.g., Co-Co is 0.98, Cr-Cr is 0.94). \* \*\*Trends:\*\* \* Tasks like 'T' (Translation), 'A' (Alignment), and 'P' (Paraphrasing) show high transferability (green hues) across most evaluation tasks, particularly toward 'Co' (Compositionality) and 'Cr' (Critical Thinking). \* Task 'D' (Detection) shows very poor transferability to other tasks, with values as low as 0.00 and 0.02, indicated by deep red cells. \* The matrix is asymmetric; for example, fine-tuning on 'P' improves 'D' by 0.64, but fine-tuning on 'D' only improves 'P' by 0.08. \*\*Panel b) Correlation Scatter Plot\*\* This panel is a scatter plot showing the relationship between model similarity and task improvement. \* \*\*X-axis:\*\* Labeled "CKA between models trained on only X vs only Y," ranging from 0.4 to 1.0. CKA (Centered Kernel Alignment) is a measure of representational similarity between neural networks. \* \*\*Y-axis:\*\* Labeled "Improvement on Y by training on X," ranging from 0.0 to 1.0. \* \*\*Data Points:\*\* Approximately 30 data points are plotted, color-coded to correspond to different task pairs. The points are scattered but show a general upward trend. \* \*\*Statistics and Annotations:\*\* \* A black dotted linear regression line is drawn through the data. \* The plot includes statistical annotations: $R^2 = 0.188$ and $p = 0.017$. \* \*\*Key Insight:\*\* There is a statistically significant positive correlation between the representational similarity of models trained on individual tasks (X and Y) and the degree to which training on task X improves performance on task Y. This suggests that tasks requiring similar internal representations are more likely to exhibit positive transfer.](figures/Figure5-1.png)
*Figure 5: Fine-tuning generalization and its correlation with representational similarity. (a) Generalization matrix (averaged over 4 seeds; see App. Fig. 16 for individual seeds): each row is a model that integrated Atlantis via one task; columns show normalized improvement on Atlantis queries for each task (see App. D.1 for metric details). (b) For each task pair (X, Y), we plot the single-task CKA between X and Y against the normalized improvement on task Y after fine-tuning on task X (see App. Fig. 15 for annotated version).*

Preprint
DIVERGENT TASKS HARM ENTITY INTEGRATION VIA FINE-TUNING
In the previous section we observed how multi-task pretraining yields shared representations for
different tasks. In this section, we investigate generalization properties of fine-tuning on top of such
representations. However, unlike most fine-tuning studies which focus on changing model behavior
in a certain way and evaluate generalization across entities, we study the inverse: fine-tuning an
entity into the model and evaluate generalization across tasks. To this end, we use the 7-task model
trained in the previous section (Fig. 4).
As mentioned in Sec. 3, we introduce 100 Atlantis cities to the world and fine-tune on data containing Atlantis to probe for generalization. We emphasize that the introduction of Atlantis
cities keeps the original dataset fully consistent with the world.
Moreover, task operations on
Atlantis cities are well-defined in the same framework. If the model learned the true data generation process with properly factored representations, it should be able to integrate Atlantis
seamlessly. If not, we suspect either the representations are fractured (Kumar et al., 2025) or gradient descent cannot trigger the right representational updates (Kumar et al., 2022).


Figure 5: Fine-tuning generalization and its correlation with representational similarity. (a) Generalization matrix (averaged over 4 seeds; see App. Fig. 16
for individual seeds): each row is a model that integrated Atlantis via one task; columns show normalized improvement on Atlantis queries for each task
(see App. D.1 for metric details). (b) For each task pair
(X, Y), we plot the single-task CKA between X and
Y against the normalized improvement on task Y after
fine-tuning on task X (see App. Fig. 15 for annotated
version).
Result 1:
Pretraining Phase Representational Alignment Predicts FineTuning Generalization Despite Joint
Pretraining
We first address a simple
question: when fine-tuning on Atlantis
cities for a single task (e.g., distance),
should we expect the model to automatically generalize to using Atlantis for
all other tasks?
To answer this, we fine-tune on 100k
examples of a single task that include
Atlantis cities, mixed with original
pretraining data to avoid catastrophic forgetting and a small multi-task elicitation
set (see App. C.3 for details).
The resulting generalization matrix is
shown in Fig. 5(a).
This matrix reveals rich phenomenology:
some tasks
like distance show no cross-task generalization (Atlantis remains usable only
for that task), while angle triggers significant generalization across all tasks. Intriguingly, we observe an apparent inverse relationship: tasks that efficiently trigger cross-task generalization of new entities are often those that don’t easily benefit from other tasks’ fine-tuning,
though this relationship is noisy.
Unexpectedly, we find that generalization performance correlates with the CKA values from singletask pretraining (Result 2 of Sec. 4). This is puzzling: the CKA values come from models trained
from scratch on individual tasks, yet they partially predict fine-tuning behavior of a model pretrained
on all tasks jointly (Fig. 5b). If the multi-task model truly uses unified representations for cities, why
would single-task representational properties matter?
For clarity, we define two terms: Divergent tasks are tasks which have low CKA compared to others
when trained in isolation (in our case the distance task). Hidden spaces are representation spaces
not surfaced by PCA or probing but used by divergent tasks.
We hypothesize:
“Even though models develop joint world representations which converge in
multi-task pretraining, gradient descent on divergent tasks might fail to act on
these shared representations during fine-tuning, instead utilizing hidden spaces
that don’t propagate updates across tasks.”


![Figure6-1: This multi-panel figure analyzes the effects of multi-task fine-tuning on model performance, specifically focusing on the concepts of "Synergy" and "Interference" across different spatial reasoning tasks. \*\*Panel (a): Heatmap of Task Interactions\*\* This panel displays a heatmap where the y-axis lists pairs of "Fine-Tuning Tasks" and the x-axis lists individual "Evaluation Tasks." The tasks are abbreviated as: D (Distance), T (Time), A (Azimuth), Co (Coordinates), I (Island), P (Population), and Cr (Country). \* \*\*Color Scale:\*\* A diverging color bar at the bottom indicates that blue represents "Synergy" (positive values, improved performance) and red/orange represents "Interference" (negative values, degraded performance). \* \*\*Key Observation:\*\* Rows involving the "Distance" task (D, highlighted in red text on the y-axis) show significant red blocks, particularly when evaluated on tasks like T, A, Co, I, P, and Cr. This indicates that fine-tuning on Distance often interferes with the model's ability to perform other spatial tasks. Conversely, many other task pairs (e.g., T,P or A,P) show blue cells, indicating synergistic performance gains. Small "T" superscripts in cells denote statistically significant results. \*\*Panel (b): Latent Space Visualizations\*\* This panel provides four scatter plots visualizing the model's internal representations (latent space). \* \*\*Top and Bottom Insets:\*\* These show 3D-like projections of data points. Red arrows point to specific clusters. The top inset shows a more dispersed, disorganized cluster compared to the bottom inset, which appears more structured. \* \*\*Middle Two Plots:\*\* These are 2D coordinate plots (x-axis from -750 to 250, y-axis from 100 to 500). Black 'x' marks represent "Ground Truth" locations, while colored dots represent "Reconstructed" locations. \* \*\*Insight:\*\* The top 2D plot (associated with a task pair showing interference) shows reconstructed points (orange/red) that are widely scattered and far from the black ground truth 'x' marks. The bottom 2D plot (associated with synergy) shows reconstructed points that are much more tightly clustered and closer to the ground truth, indicating higher accuracy. \*\*Panel (c): Histogram of Deviation from Predictions\*\* This histogram shows the distribution of "Deviation from predictions" across all task combinations. \* \*\*X-axis:\*\* Ranges from -0.5 to 0.5. Values greater than 0 (blue bars) indicate "Synergy," while values less than 0 (red bars) indicate "Interference." \* \*\*Y-axis:\*\* Count of task combinations. \* \*\*Trend:\*\* The distribution is bimodal. There is a large peak in the blue region (around 0.05 to 0.15), showing that most task combinations result in synergy. However, a smaller, distinct cluster of red bars exists between -0.1 and -0.4, representing the significant interference caused by specific task pairings (primarily those involving Distance). \*\*Panel (d): Histogram of Reconstruction Error\*\* This plot compares reconstruction errors across different data subsets on a logarithmic x-axis. \* \*\*X-axis:\*\* Reconstruction Error (log scale from $10^0$ to $10^3$). \* \*\*Y-axis:\*\* Count. \* \*\*Data Series:\*\* \* \*\*Yellow (Non-Atlantis Cities):\*\* Shows the lowest error, peaking around 20. A solid green vertical line marks "Atlantis In Pretraining" error level, which aligns with this group. \* \*\*Blue (Distance Task Not Included):\*\* Shows moderate error, peaking around 100-200. \* \*\*Red (Distance Task Included):\*\* Shows the highest error, with a significant peak near 500-600. \* \*\*Insight:\*\* This panel quantifies the "Interference" seen in panel (a). Including the Distance task during fine-tuning drastically increases the reconstruction error (shifts the distribution to the right) compared to when it is excluded. \*\*Summary of Insights:\*\* The figure demonstrates that while multi-task fine-tuning generally leads to synergistic performance improvements in spatial reasoning, certain tasks—specifically "Distance" (D)—act as strong interferers. When Distance is included in fine-tuning, the model's internal representations of space degrade, leading to significantly higher reconstruction errors and poorer performance on related evaluation tasks.](figures/Figure6-1.png)
*Figure 6: Divergent tasks harm multi-task fine-tuning and disrupt representational integration. (a) Deviation from best-teacher expectation for 21 two-task models (rows) across 7 evaluation tasks (columns), computed in normalized improvement space (see App. D.1); “red horizontal bands” show distance task combinations degrade performance below single-task baselines. (b) Representation visualization and linear probe reconstruction of Atlantis. (c) Histogram of deviation values: models including distance vs. not. (d) Linear probe Atlantis coordinate reconstruction error for models with distance, without distance, and baseline on pretraining cities; green vertical line indicates performance when Atlantis is part of pretraining. 3D visualizations: link .*

Preprint
Our question is then two-part:


## 1. To what extent do divergent tasks affect fine-tuning generalization?



## 2. Will gradient descent on divergent tasks fail to merge fine-tuning introduced concepts to


the original representation manifold?
Result 2: Divergent Tasks Catastrophically Harm Generalization
To investigate how divergent
tasks affect generalization, we move from single-task to multi-task fine-tuning settings. First, we
introduce a simple heuristic model: fine-tuning on a concatenated dataset {D1, D2, ..., Dn} (which
do not provide conflicting supervision) should combine their individual effects. Specifically, when
concatenating and shuffling all fine-tuning data to avoid sequential learning effects like catastrophic
forgetting (McCloskey & Cohen, 1989), we expect the improvement Impi on task i after training on
tasks j and k to follow a best-teacher model:
Impi(Dj ∪Dk) = max(Impi(Dj), Impi(Dk))
(1)
To test this hypothesis, we fine-tuned the 7-task model on all
 7

= 21 possible two-task combinations. Fig. 6(a,c) shows the deviation from our best-teacher expectation (averaged over 4 seeds; see
App. Fig. 17 for raw improvements and App. Fig. 18 for individual seeds). Strikingly, we observe
“red horizontal bands”, models that not only fail to benefit from multi-task training but actually perform worse than their best single-task component. Notably, all these degraded performance bands
involve the distance task. Fig. 6(c) quantifies this: when we split the deviation values into models with and without distance, we consistently observe lower-than-expected performance when
the divergent task is included. This confirms that divergent tasks (those with low single-task CKA)
actively harm fine-tuning generalization rather than simply failing to contribute. We next examine
how this manifests in the learned representations.


Figure 6: Divergent tasks harm multi-task fine-tuning and disrupt representational integration. (a) Deviation from best-teacher expectation for 21 two-task models (rows) across 7 evaluation
tasks (columns), computed in normalized improvement space (see App. D.1); “red horizontal bands”
show distance task combinations degrade performance below single-task baselines. (b) Representation visualization and linear probe reconstruction of Atlantis. (c) Histogram of deviation
values: models including distance vs. not. (d) Linear probe Atlantis coordinate reconstruction error for models with distance, without distance, and baseline on pretraining cities;
green vertical line indicates performance when Atlantis is part of pretraining. 3D visualizations:
link .

Preprint
Result 3: Divergent Tasks Disrupt Representational Integration of New Entities
Having
shown that divergent tasks harm generalization (Question 1), we now address Question 2: does
gradient descent on divergent tasks fail to merge new entities into the representation manifold?
We take two exemplars from the 21 fine-tuning runs: one without distance that generalized well
(angle + compass), and one with distance that was harmed (distance + perimeter).
We first train a linear probe on a subset of all cities including Atlantis; these reconstructions
are shown in Fig. 6(b) (top and bottom panels). In the well-integrated case, Atlantis cities lie
within the world data manifold. In the ill-integrated case, Atlantis cities are off the manifold.
While this difference appears subtle in 2D projections, the effect is dramatic in 3D—we strongly
encourage readers to explore our interactive visualizations . Next, we train a linear probe on 4000
non-Atlantis cities and apply it to Atlantis representations (middle panels). In the wellintegrated case, Atlantis cities (red-orange) are relatively well reconstructed compared to ground
truth (black crosses); in the ill-integrated case, reconstruction fails completely.
We quantify this effect in Fig. 6(d), showing histograms of absolute coordinate reconstruction error.
When Atlantis is integrated via fine-tuning partially on divergent task data (red), reconstruction
errors are nearly an order of magnitude larger than when integrated via purely non-divergent tasks
(blue). For reference, non-Atlantis cities (yellow, still held out from probe training) show low
reconstruction error as expected. One might hypothesize that Atlantis’s location in the middle of
the ocean creates inherently difficult geometry. To test this, we pretrained a model with Atlantis
included from the start (green line). In this case, Atlantis cities are reconstructed as well as any
other city, confirming that the integration failure stems from divergent task fine-tuning dynamics
rather than geographic peculiarity.
This suggests that divergent tasks cause optimization to encode new entities in hidden spaces rather
than integrating them into the existing world manifold, explaining their failure to support cross-task
generalization.
We emphasize that our findings are correlational: we do not claim that interventions to increase
single-task CKA would necessarily improve fine-tuning generalization. Rather, we identify representational divergence as a diagnostic marker for tasks that will harm multi-task fine-tuning performance.
Putting these results together: single-task representational divergence weakly predicts fine-tuning
generalization even after joint pretraining, and the most divergent task (distance) actively harms
integration of new entities. This raises a hypothesis: certain task-architecture pairings may have
intrinsic properties that induce gradient dynamics bypassing shared representations, causing updates
in hidden subspaces that harm generalization, even when the network uses unified representations
for the forward pass.


### DISCUSSION


Continual learning and world models. For truly general intelligence, internal world models should
not only represent current state but adapt consistently when the world changes. Such adaptation is
non-trivial: a single change can require cascading updates across tasks. Recent language models
sidestep persistent adaptation via in-context learning, forming task-specific representations on the
fly (Brown et al., 2020; Park et al., 2024a; Li et al., 2025b). However, fine-tuning consistently
underperforms ICL for knowledge integration (Lampinen et al., 2025; Park et al., 2025). Our study
grounds these questions in a controlled setting where we can measure whether gradient descent
achieves consistent integration of new entities into existing representations.
Dynamics of representations. Most recent work on neural representations examines pretrained
networks or their formation during a single pretraining run. There is growing interest in how representations change during adaptation, both at inference (Park et al., 2024a; Li et al., 2025b; Shai et al.,
2025; Lubana et al., 2025; Bigelow et al., 2025) and during fine-tuning (Wang et al., 2025; Minder
et al., 2025; Casademunt et al., 2025). To study representational adaptation rigorously, one must
define both an updatable world and how updates to it propagate into training data. Our framework
provides exactly this: introducing Atlantis defines how representations should update across all
tasks.

Preprint
Forward and backward modularity. Our results highlight a distinction that is often overlooked:
modularity in the forward pass does not imply modularity in the backward pass. Multi-task training
produces clean, structured representations that can be easily decoded into world coordinates, yet
these world models can be fractured and partial when it comes to adaptation. Gradient descent
may not respect the forward-pass modularity when updating weights: fine-tuning on divergent tasks
routes updates through pathways that bypass the shared world manifold, encoding new entities in
task-specific subspaces.
Future work. Understanding the mechanistic basis of task divergence is an important open question. If divergence is a property of task-architecture pairing rather than learned weights, it may be
predictable from task structure and gradient geometry alone, enabling identification of harmful tasks
before training.
Limitations. We study representation formation in a controlled synthetic setting with small-scale
models; generalization to large-scale natural settings remains unclear. We identify divergence as
a diagnostic marker but do not reveal underlying mechanisms. Our PRH claims are partial, as we
study only a single architecture and modality.


### CONCLUSION


We introduced a World–Data–Model framework that separates the underlying world from the data
generation process, enabling controlled study of how representations form and adapt. Crucially, this
separation allows defining consistent world updates (adding new entities that integrate seamlessly
across all tasks), providing clear expectations for what proper world representations should support.
Using this framework, we first showed that multi-task training drives representational convergence:
models trained on disjoint task sets develop aligned representations, providing partial evidence for
the Multitask Scaling Hypothesis. However, this convergence does not guarantee consistent adaptation: certain “divergent” tasks actively harm the integration of new entities during fine-tuning,
encoding them in hidden spaces rather than the shared world manifold. This highlights a distinction between forward and backward modularity: clean, structured representations do not necessarily
adapt cleanly to new information.

Preprint


### USE OF LARGE LANGUAGE MODELS


Large language models were used for:
• Assistance in finding related papers during literature review.
• Boilerplate code for research.
• Refining the language of the manuscript.


### REPRODUCIBILITY STATEMENT


All data generation, model training and analysis were carefully tracked with configuration files to
ensure reproducibility. All random seeds for dataset generation and model training were tracked
as well (all set to 42). All code, data and analysis results are openly available. Furthermore, the
authors have open sourced the entire research process including the process on converging to the set
of experiments presented in the paper.


### REFERENCES


Alessandro Achille, Matteo Rovere, and Stefano Soatto. Critical learning periods in deep neural
networks, 2019. URL https://arxiv.org/abs/1711.08856.
Armen Aghajanyan, Anchit Gupta, Akshat Shrivastava, Xilun Chen, Luke Zettlemoyer, and Sonal
Gupta. Muppet: Massive multi-task representations with pre-finetuning, 2021. URL https:
//arxiv.org/abs/2101.11038.
Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.1, knowledge storage and
extraction. arXiv preprint arXiv:2309.14316, 2023a.
Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 1, learning hierarchical language structures. ArXiv e-prints, abs/2305.13673, May, 2023b.
Anthropic AI.
Towards Monosemanticity:
Decomposing Language Models With Dictionary
Learning,
2023.
https://transformer-circuits.pub/2023/
monosemantic-features.
Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and
Neel Nanda.
Refusal in language models is mediated by a single direction.
arXiv preprint
arXiv:2406.11717, 2024.
Gregor Bachmann and Vaishnavh Nagarajan. The pitfalls of next-token prediction, 2025. URL
https://arxiv.org/abs/2403.06963.
Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new
perspectives, 2014. URL https://arxiv.org/abs/1206.5538.
Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. The reversal curse: Llms trained on ”a is b” fail to learn ”b is a”, 2024.
URL https://arxiv.org/abs/2309.12288.
Jan Betley, Daniel Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Mart´ın Soto, Nathan
Labenz, and Owain Evans. Emergent misalignment: Narrow finetuning can produce broadly
misaligned llms, 2025. URL https://arxiv.org/abs/2502.17424.
Eric Bigelow, Daniel Wurgaft, YingQiao Wang, Noah Goodman, Tomer Ullman, Hidenori Tanaka,
and Ekdeep Singh Lubana. Belief dynamics reveal the dual nature of in-context learning and
activation steering, 2025. URL https://arxiv.org/abs/2511.00617.
Michael M. Bronstein, Joan Bruna, Taco Cohen, and Petar Veliˇckovi´c.
Geometric deep learning: Grids, groups, graphs, geodesics, and gauges, 2021. URL https://arxiv.org/abs/
2104.13478.

Preprint
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
Rich Caruana. Multitask learning. Machine learning, 28(1):41–75, 1997.
Helena Casademunt, Caden Juang, Adam Karvonen, Samuel Marks, Senthooran Rajamanoharan,
and Neel Nanda. Steering out-of-distribution generalization with concept ablation fine-tuning,


## 2025. URL https://arxiv.org/abs/2507.16795.


Stephanie C. Y. Chan, Adam Santoro, Andrew K. Lampinen, Jane X. Wang, Aaditya Singh, Pierre H.
Richemond, Jay McClelland, and Felix Hill. Data distributional properties drive emergent incontext learning in transformers, 2022. URL https://arxiv.org/abs/2205.05055.
Taco S. Cohen and Max Welling. Group equivariant convolutional networks, 2016. URL https:
//arxiv.org/abs/1602.07576.
R´obert Csord´as, Christopher Potts, Christopher D Manning, and Atticus Geiger. Recurrent neural
networks learn to store and generate sequences using non-linear representations. arXiv preprint
arXiv:2408.10920, 2024.
Can Demircan, Tankred Saanum, Akshay K. Jagadish, Marcel Binz, and Eric Schulz. Sparse autoencoders reveal temporal difference learning in large language models, 2024. URL https:
//arxiv.org/abs/2410.01280.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
Shibhansh Dohare, J. Fernando Hernandez-Garcia, Parash Rahman, A. Rupam Mahmood, and
Richard S. Sutton.
Maintaining plasticity in deep continual learning, 2024.
URL https:
//arxiv.org/abs/2306.13812.
Joshua Engels, Isaac Liao, Eric J. Michaud, Wes Gurnee, and Max Tegmark. Not all language model
features are linear, 2024. URL https://arxiv.org/abs/2405.14860.
Stephanie Fu, Tyler Bonnen, Devin Guillory, and Trevor Darrell. Hidden in plain sight: Vlms overlook their visual representations, 2025. URL https://arxiv.org/abs/2506.08008.
Kunihiko Fukushima. Neocognitron: A self-organizing neural network model for a mechanism of
pattern recognition unaffected by shift in position. Biological cybernetics, 36(4):193–202, 1980.
Xuyang Ge, Wentao Shu, Jiaxing Wu, Yunhua Zhou, Zhengfu He, and Xipeng Qiu. Evolution of
concepts in language model pre-training, 2025. URL https://arxiv.org/abs/2509.
17196.
Pulkit Gopalani and Wei Hu. What happens during the loss plateau? understanding abrupt learning
in transformers, 2025. URL https://arxiv.org/abs/2506.13688.
Wes Gurnee and Max Tegmark.
Language models represent space and time.
arXiv preprint
arXiv:2310.02207, 2023.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015.
Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick,
Shakir Mohamed, and Alexander Lerchner.
beta-vae: Learning basic visual concepts with a
constrained variational framework. In Proc. Int. Conf. on Learning Representations (ICLR), 2017.
Sai Sumedh R. Hindupur, Ekdeep Singh Lubana, Thomas Fel, and Demba Ba. Projecting assumptions: The duality between sparse autoencoders and concept geometry, 2025. URL https:
//arxiv.org/abs/2503.01822.
David T. Hoffmann, Simon Schrodi, Jelena Bratuli´c, Nadine Behrmann, Volker Fischer, and Thomas
Brox. Eureka-moments in transformers: Multi-step tasks reveal softmax induced optimization
problems, 2024. URL https://arxiv.org/abs/2310.12956.

Preprint
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. URL https:
//arxiv.org/abs/2106.09685.
David H Hubel and Torsten N Wiesel. Receptive fields, binocular interaction and functional architecture in the cat’s visual cortex. The Journal of physiology, 160(1):106, 1962.
Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. The platonic representation
hypothesis, 2024. URL https://arxiv.org/abs/2405.07987.
Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt,
Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic, 2023. URL https:
//arxiv.org/abs/2212.04089.
Samyak Jain, Robert Kirk, Ekdeep Singh Lubana, Robert P Dick, Hidenori Tanaka, Edward Grefenstette, Tim Rockt¨aschel, and David Scott Krueger. Mechanistically analyzing the effects of finetuning on procedurally defined tasks. arXiv preprint arXiv:2311.12786, 2023.
Jaeyeon Kim, Sehyun Kwon, Joo Young Choi, Jongho Park, Jaewoong Cho, Jason D. Lee, and
Ernest K. Ryu. Task diversity shortens the icl plateau, 2025. URL https://arxiv.org/
abs/2410.05448.
Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of Neural
Network Representations Revisited. In Proc. of the 36th Proc. Int. Conf. on Machine Learning
(ICML), Proc. of Machine Learning Research. PMLR, 09–15 Jun 2019.
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.
Akarsh Kumar, Jeff Clune, Joel Lehman, and Kenneth O. Stanley. Questioning representational
optimism in deep learning: The fractured entangled representation hypothesis, 2025.
URL
https://arxiv.org/abs/2505.11581.
Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, and Percy Liang. Fine-tuning can
distort pretrained features and underperform out-of-distribution, 2022. URL https://arxiv.
org/abs/2202.10054.
Andrew K. Lampinen, Arslan Chaudhry, Stephanie C. Y. Chan, Cody Wild, Diane Wan, Alex Ku,
J¨org Bornschein, Razvan Pascanu, Murray Shanahan, and James L. McClelland. On the generalization of language models from in-context learning and finetuning: a controlled study, 2025.
URL https://arxiv.org/abs/2505.00661.
Andrew Lee, Xiaoyan Bai, Itamar Pres, Martin Wattenberg, Jonathan K Kummerfeld, and Rada
Mihalcea.
A mechanistic understanding of alignment algorithms: A case study on dpo and
toxicity. In Forty-first International Conference on Machine Learning, 2024. URL https:
//arxiv.org/abs/2401.01967.
Andrew Lee, Lihao Sun, Chris Wendler, Fernanda Vi´egas, and Martin Wattenberg. The geometry of
self-verification in a task-specific reasoning model, 2025. URL https://arxiv.org/abs/
2504.14379.
Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt
tuning, 2021. URL https://arxiv.org/abs/2104.08691.
Kenneth Li, Aspen K Hopkins, David Bau, Fernanda Vi´egas, Hanspeter Pfister, and Martin Wattenberg. Emergent world representations: Exploring a sequence model trained on a synthetic task.
In The Eleventh International Conference on Learning Representations, 2022.
Melody Zixuan Li, Kumar Krishna Agrawal, Arna Ghosh, Komal Kumar Teru, Guillaume Lajoie, and Blake Aaron Richards.
Tracing the representation geometry of language models
from pretraining to post-training. In High-dimensional Learning Dynamics 2025, 2025a. URL
https://openreview.net/forum?id=9nKmDLXg9v.

Preprint
Yuxuan Li, Declan Campbell, Stephanie C. Y. Chan, and Andrew Kyle Lampinen. Just-in-time
and distributed task representations in language models, 2025b. URL https://arxiv.org/
abs/2509.04466.
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019.
Ekdeep Singh Lubana, Can Rager, Sai Sumedh R. Hindupur, Valerie Costa, Greta Tuckute, Oam
Patel, Sonia Krishna Murthy, Thomas Fel, Daniel Wurgaft, Eric J. Bigelow, Johnny Lin, Demba
Ba, Martin Wattenberg, Fernanda Viegas, Melanie Weber, and Aaron Mueller. Priors in time:
Missing inductive biases for language model interpretability, 2025. URL https://arxiv.
org/abs/2511.01836.
Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, and
Sanjeev Arora. Fine-tuning language models with just forward passes, 2024. URL https:
//arxiv.org/abs/2305.17333.
Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language
model representations of true/false datasets, 2024. URL https://arxiv.org/abs/2310.
06824.
Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The
sequential learning problem. In Psychology of learning and motivation, volume 24, pp. 109–165.
Elsevier, 1989.
Abhinav Menon, Manish Shrivastava, David Krueger, and Ekdeep Singh Lubana.
Analyzing
(in)abilities of saes via formal languages, 2025. URL https://arxiv.org/abs/2410.
11767.
Eric J Michaud, Ziming Liu, Uzay Girit, and Max Tegmark. The quantization model of neural
scaling. arXiv preprint arXiv:2303.13506, 2023.
Julian Minder, Cl´ement Dumas, Caden Juang, Bilal Chugtai, and Neel Nanda. Overcoming sparsity
artifacts in crosscoders to interpret chat-tuning, 2025.
URL https://arxiv.org/abs/
2504.02922.
Andrei Mircea, Supriyo Chakraborty, Nima Chitsazan, Milind Naphade, Sambit Sahu, Irina Rish,
and Ekaterina Lobacheva. Training dynamics underlying language model scaling laws: Loss
deceleration and zero-sum learning, 2025. URL https://arxiv.org/abs/2506.05447.
Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability, 2023a. URL https://arxiv.org/abs/
2301.05217.
Neel Nanda, Andrew Lee, and Martin Wattenberg.
Emergent linear representations in world
models of self-supervised sequence models.
In Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pp. 16–30, 2023b. URL https:
//arxiv.org/abs/2309.00941.
Kento Nishi, Maya Okawa, Rahul Ramesh, Mikail Khona, Ekdeep Singh Lubana, and Hidenori
Tanaka. Representation shattering in transformers: A synthetic study with knowledge editing.
arXiv preprint arXiv:2410.17194, 2024.
Maya Okawa, Ekdeep Singh Lubana, Robert P. Dick, and Hidenori Tanaka. Compositional abilities
emerge multiplicatively: Exploring diffusion models on a synthetic task, 2024.
Chris Olah, Alexander Mordvintsev, and Ludwig Schubert. Feature visualization. Distill, 2017. doi:
10.23915/distill.00007. https://distill.pub/2017/feature-visualization.
OpenDataSoft
/
GeoNames.
Geonames
–
all
cities
with
a
population
¿
1000.
https://public.opendatasoft.com/explore/dataset/
geonames-all-cities-with-a-population-1000, 2025. Accessed: 2025.

Preprint
Core Francisco Park, Andrew Lee, Ekdeep Singh Lubana, Yongyi Yang, Maya Okawa, Kento Nishi,
Martin Wattenberg, and Hidenori Tanaka. Iclr: In-context learning of representations, 2024a.
URL https://arxiv.org/abs/2501.00070.
Core Francisco Park, Ekdeep Singh Lubana, Itamar Pres, and Hidenori Tanaka. Competition dynamics shape algorithmic phases of in-context learning. arXiv preprint arXiv:2412.01003, 2024b.
Core Francisco Park, Maya Okawa, Andrew Lee, Ekdeep Singh Lubana, and Hidenori Tanaka.
Emergence of hidden capabilities: Exploring learning dynamics in concept space, 2024c. URL
https://arxiv.org/abs/2406.19370.
Core Francisco Park, Zechen Zhang, and Hidenori Tanaka. New News: System-2 fine-tuning for robust integration of new knowledge, 2025. URL https://arxiv.org/abs/2505.01812.
Michael Pearce, Elana Simon, Michael Byun, and Daniel Balsam. Finding the tree of life in evo 2.
Goodfire Research, August 2025. Correspondence to michael@goodfire.ai.
Mohammad Pezeshki, Oumar Kaba, Yoshua Bengio, Aaron C Courville, Doina Precup, and Guillaume Lajoie. Gradient starvation: A learning proclivity in neural networks. Adv. in Neural
Information Processing Systems (NeurIPS), 2021.
Tian Qin, Core Francisco Park, Mujin Kwun, Aaron Walsman, Eran Malach, Nikhil Anand, Hidenori
Tanaka, and David Alvarez-Melis. Decomposing elements of problem solving: What ”math” does
rl teach?, 2025. URL https://arxiv.org/abs/2505.22756.
Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training, 2018.
Allan Ravent´os, Mansheej Paul, Feng Chen, and Surya Ganguli. Pretraining task diversity and the
emergence of non-bayesian in-context learning for regression, 2023. URL https://arxiv.
org/abs/2306.15063.
Gautam Reddy. The mechanistic basis of data dependence and abrupt learning in an in-context
classification task, 2023. URL https://arxiv.org/abs/2312.03002.
Frank Rosenblatt. The perceptron: a probabilistic model for information storage and organization
in the brain. Psychological review, 65(6):386, 1958.
David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by backpropagating errors. nature, 323(6088):533–536, 1986.
Harshay Shah, Kaustav Tamuly, Aditi Raghunathan, Prateek Jain, and Praneeth Netrapalli. The
pitfalls of simplicity bias in neural networks, 2020. URL https://arxiv.org/abs/2006.
07710.
Adam S. Shai, Sarah E. Marzen, Lucas Teixeira, Alexander Gietelink Oldenziel, and Paul M.
Riechers.
Transformers represent belief state geometry in their residual stream, 2025.
URL
https://arxiv.org/abs/2405.15943.
Aaditya K. Singh, Ted Moskovitz, Felix Hill, Stephanie C. Y. Chan, and Andrew M. Saxe. What
needs to go right for an induction head? a mechanistic study of in-context learning circuits and
their formation, 2024. URL https://arxiv.org/abs/2404.07129.
Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen,
Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L
Turner, Callum McDougall, Monte MacDiarmid, C. Daniel Freeman, Theodore R. Sumers,
Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, and Tom Henighan.
Scaling monosemanticity:
Extracting interpretable features from claude 3 sonnet.
Transformer Circuits Thread, 2024.
URL https://transformer-circuits.pub/2024/
scaling-monosemanticity/index.html.
Johannes Treutlein, Dami Choi, Jan Betley, Samuel Marks, Cem Anil, Roger Grosse, and Owain
Evans. Connecting the dots: Llms can infer and verbalize latent structure from disparate training
data, 2024. URL https://arxiv.org/abs/2406.14546.

Preprint
Keyon Vafa, Peter G. Chang, Ashesh Rambachan, and Sendhil Mullainathan. What has a foundation
model found? using inductive bias to probe for world models, 2025. URL https://arxiv.
org/abs/2507.06952.
Atticus Wang, Joshua Engels, Oliver Clive-Griffin, Senthooran Rajamanoharan, and Neel Nanda.
Simple mechanistic explanations for out-of-context reasoning, 2025. URL https://arxiv.
org/abs/2507.08218.
Jake Ward, Chuqiao Lin, Constantin Venhoff, and Neel Nanda. Reasoning-finetuning repurposes latent representations in base models, 2025. URL https://arxiv.org/abs/2507.12638.
Maurice Weiler and Gabriele Cesa. General e(2)-equivariant steerable cnns, 2021. URL https:
//arxiv.org/abs/1911.08251.
Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D. Manning, and Christopher Potts. Reft: Representation finetuning for language models, 2024. URL
https://arxiv.org/abs/2404.03592.
Daniel Wurgaft, Ekdeep Singh Lubana, Core Francisco Park, Hidenori Tanaka, Gautam Reddy,
and Noah D. Goodman. In-context learning strategies emerge rationally, 2025. URL https:
//arxiv.org/abs/2506.17859.
Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context
learning as implicit bayesian inference. arXiv preprint arXiv:2111.02080, 2021.
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
Dayiheng Liu, Fei Huang, Haoran Wei, et al.


Qwen2. 5 technical report.
arXiv preprint
arXiv:2412.15115, 2024.
Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao
Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the
base model?, 2025. URL https://arxiv.org/abs/2504.13837.
Shiyang Zhang, Aakash Patel, Syed A Rizvi, Nianchen Liu, Sizhuang He, Amin Karbasi, Emanuele
Zappala, and David van Dijk. Intelligence at the edge of chaos, 2025. URL https://arxiv.
org/abs/2410.02536.
Rosie Zhao, Alexandru Meterez, Sham Kakade, Cengiz Pehlevan, Samy Jelassi, and Eran Malach.
Echo chamber: Rl post-training amplifies behaviors learned in pretraining, 2025. URL https:
//arxiv.org/abs/2504.07912.
Adam Zweiger, Jyothish Pari, Han Guo, Ekin Aky¨urek, Yoon Kim, and Pulkit Agrawal.
Selfadapting language models, 2025. URL https://arxiv.org/abs/2506.10943.


![Figure7-1: This figure presents a two-dimensional scatter plot that visualizes a global dataset, likely representing a dimensionality reduction (such as t-SNE or UMAP) of geographic or genetic data. The plot uses a Cartesian coordinate system where the data points are arranged in a manner that closely mimics a world map. ### Axes and Scale \* \*\*X-axis:\*\* Labeled "X" in a large, bold font. The scale ranges from approximately -1600 to 1800, with major tick marks every 500 units. \* \*\*Y-axis:\*\* Labeled "Y" in a large, bold font. The scale ranges from approximately -600 to 800, with major tick marks every 500 units. ### Data Distribution and Color Coding The data consists of thousands of small dots, color-coded into 13 distinct geographic regions. The spatial arrangement of these clusters reproduces the general shape of the Earth's continents: 1. \*\*North America (Yellow):\*\* Located in the upper left quadrant, spanning X values from -1300 to -700 and Y values from 100 to 550. 2. \*\*South America (Orange):\*\* Located below North America, spanning X values from -800 to -300 and Y values from -550 to 150. 3. \*\*Africa (Blue-Violet):\*\* Occupies the center-bottom area, spanning X values from -200 to 600 and Y values from -400 to 300. 4. \*\*Western Europe (Dark Red):\*\* Located in the upper center, spanning X values from -100 to 300 and Y values from 400 to 700. 5. \*\*Eastern Europe (Light Orange/Tan):\*\* Situated to the right of Western Europe, spanning X values from 200 to 1500 and Y values from 400 to 750. 6. \*\*Middle East (Pink):\*\* Located between Africa and Eastern Europe, spanning X values from 300 to 700 and Y values from 100 to 450. 7. \*\*Central Asia (Teal):\*\* A small cluster located around X=700, Y=500. 8. \*\*South Asia (Light Blue):\*\* Located below Central Asia, spanning X values from 700 to 900 and Y values from 50 to 350. 9. \*\*South East Asia (Green):\*\* Located to the right of South Asia, spanning X values from 900 to 1300 and Y values from -100 to 250. 10. \*\*China (Purple):\*\* A dense cluster in the upper right, spanning X values from 1000 to 1300 and Y values from 200 to 500. 11. \*\*Korea (Light Green):\*\* A small cluster adjacent to China at X=1250, Y=400. 12. \*\*Japan (Rose Pink):\*\* A small cluster to the right of Korea at X=1400, Y=400. 13. \*\*Oceania (Steel Blue):\*\* Scattered points in the bottom right, spanning X values from 1100 to 1800 and Y values from -500 to -100. ### Key Annotation The most prominent feature of the figure is a specific cluster of points colored in a reddish-orange hue, located in the "Atlantic Ocean" area between North America and Africa (centered around X=-400, Y=350). \* A thick, reddish-orange arrow points directly to this cluster. \* The arrow is labeled with the word \*\*"Atlantis"\*\* in a matching bold, reddish-orange font. ### Scientific Insight The figure demonstrates that the underlying data (likely genomic or linguistic) contains enough structure to reconstruct global geography through unsupervised or semi-supervised dimensionality reduction. The inclusion of the "Atlantis" cluster is a notable anomaly or a deliberate insertion, as it represents a landmass and population group that does not exist in modern geography, positioned where the mythical continent is traditionally described. This suggests the figure may be from a study involving synthetic data, simulation, or a creative exploration of data structures.](figures/Figure7-1.png)
*Figure 7: Geographic distribution of cities used in our experiments. 5,075 real-world cities plus 100 synthetic Atlantis cities (5,175 total). Cities span all continents and provide a fixed, measurable world structure. Coordinates use an equirectangular projection: x = 10×longitude, y = 10 × latitude (in degrees). The Atlantis region (Atlantic Ocean) is used for out-of-distribution testing.*

Preprint


### APPENDIX


A


### RESEARCH PROCESS


The
whole
research
process
is
available
at:
https://cfpark00.github.io/
world-rep-research-flow/
B


### 3D VISUALIZATIONS


3D visualizations are available here (Open Science Framework link).
C


### EXPERIMENTAL DETAILS


This section provides detailed information about the world, data generation process, model architecture, and training procedures used in our experiments.
C.1


### WORLD



Figure 7: Geographic distribution of cities used in our experiments. 5,075 real-world cities
plus 100 synthetic Atlantis cities (5,175 total). Cities span all continents and provide a fixed,
measurable world structure. Coordinates use an equirectangular projection: x = 10×longitude, y =
10 × latitude (in degrees). The Atlantis region (Atlantic Ocean) is used for out-of-distribution
testing.
Our experiments use a geographic world consisting of 5,075 cities extracted from the GeoNames (OpenDataSoft / GeoNames, 2025) database with population greater than 100,000. Cities
are distributed across all continents. This choice provides natural variation in density (e.g., dense
regions like India versus sparse Oceania) that creates interesting computational challenges.
While we use real city coordinates, this work studies abstract geometric reasoning rather than actual geography—we project coordinates to Euclidean space using an equirectangular projection (as
described above) and treat all tasks as pure geometry problems.
We deliberately chose a flat 2D manifold rather than a spherical globe. Our early experiments used
spherical coordinates, but we realized that regardless of the external world’s geometry, the model
must construct its own internal representation starting from random entity distributions. Given the
model’s nonlinearity, there is no fundamental reason why any particular geometry (planar, spherical, etc.) would be canonical. Our choice of planar geometry enables clean linear probing to read


![Table1-1: This table outlines seven distinct geometric or spatial reasoning tasks used in a scientific study, likely related to machine learning or spatial cognition. The table is organized into five columns: \*\*Task\*\*, \*\*Input\*\*, \*\*Output Type\*\*, \*\*Unit/Values\*\*, and \*\*Example\*\*. The tasks are as follows: 1. \*\*distance\*\*: \* \*\*Input\*\*: 2 cities. \* \*\*Output Type\*\*: Numerical. \* \*\*Unit/Values\*\*: Scaled coordinates (linear distance). \* \*\*Example\*\*: `dist(c_865, c_4879) = 769` 2. \*\*triarea\*\* (Triangle Area): \* \*\*Input\*\*: 3 cities. \* \*\*Output Type\*\*: Numerical. \* \*\*Unit/Values\*\*: Scaled coordinates squared (area). \* \*\*Example\*\*: `triarea(c_1234, c_5678, c_9012) = 45823` 3. \*\*angle\*\*: \* \*\*Input\*\*: 3 cities. \* \*\*Output Type\*\*: Numerical. \* \*\*Unit/Values\*\*: Degrees ranging from 0 to 180. \* \*\*Example\*\*: `angle(c_2345, c_6789, c_123) = 97` 4. \*\*compass\*\*: \* \*\*Input\*\*: 2 cities. \* \*\*Output Type\*\*: Categorical. \* \*\*Unit/Values\*\*: 8 cardinal and intercardinal directions (e.g., N, NE, E, SE, S, SW, W, NW). \* \*\*Example\*\*: `compass(c_1234, c_5678) = NE` 5. \*\*inside\*\*: \* \*\*Input\*\*: 1 city plus $n$ cities (defining a boundary). \* \*\*Output Type\*\*: Categorical (Boolean). \* \*\*Unit/Values\*\*: TRUE/FALSE. \* \*\*Example\*\*: `inside(c_9012; c_3456, ...) = FALSE` 6. \*\*perimeter\*\*: \* \*\*Input\*\*: $n$ cities. \* \*\*Output Type\*\*: Numerical. \* \*\*Unit/Values\*\*: Scaled coordinates (sum of distances). \* \*\*Example\*\*: `perimeter(c_4567, c_8901, ...) = 2856` 7. \*\*crossing\*\*: \* \*\*Input\*\*: 4 cities (defining two line segments). \* \*\*Output Type\*\*: Categorical (Boolean). \* \*\*Unit/Values\*\*: TRUE/FALSE. \* \*\*Example\*\*: `cross(c_2345, c_6789; c_123, c_4567) = TRUE` \*\*Key Insights\*\*: The table demonstrates a variety of spatial reasoning challenges ranging from simple distance calculations to complex topological relationships (inside, crossing). The tasks utilize both continuous numerical outputs and discrete categorical classifications. The use of "scaled coordinates" suggests the data is normalized or mapped to a specific coordinate system rather than using raw geographic coordinates like latitude and longitude. The city identifiers (e.g., `c_865`) indicate a dataset where locations are treated as discrete points in a 2D space.](figures/Table1-1.png)
*Table 1: Summary of 7 geometric tasks. Numerical outputs are integers; “scaled coords” refers to the ×10 coordinate system (Sec. C.1). Categorical tasks have discrete outputs: compass uses 8 cardinal directions (N, NE, E, SE, S, SW, W, NW), while inside and crossing are binary. The inside task tests if the first city lies within the convex hull of the remaining cities; crossing tests if line segment (c1, c2) intersects segment (c3, c4).*

Preprint
out world representations, whereas extracting nonlinear manifold structure remains an open challenge (Engels et al., 2024; Csord´as et al., 2024). While geometric deep learning (Bronstein et al.,
2021) studies the interaction between data geometry and model computation, our focus is on general
sequence modeling rather than geometry-aware architectures.
Additionally, we introduce 100 synthetic Atlantis cities positioned in the Atlantic Ocean, centered at (longitude −35◦, latitude 35◦) and following a Gaussian distribution with standard deviation of 3◦. These synthetic cities enable controlled out-of-distribution experiments, as models never
observe Atlantis during pretraining but must generalize to it during evaluation. City IDs are randomly assigned from the range [0, 9999], creating a sparse identifier space that models must learn
to map to coordinates. All coordinates are stored as integers (after the ×10 scaling), eliminating
floating-point precision issues.
C.2


### DATA GENERATION PROCESS


Tasks
We implement 7 geometric tasks that operate on city coordinates. All tasks use a consistent
format: task(arguments)=answer, where city IDs are prefixed with c . Numerical outputs
(distance, area, angle, perimeter) are rounded to integers. Table 1 summarizes the tasks.
Task
Input
Output Type
Unit/Values
Example
distance
2 cities
Numerical
Scaled coords
dist(c 865,c 4879)=769
triarea
3 cities
Numerical
Scaled coords2
triarea(c 1234,c 5678,c 9012)=45823
angle
3 cities
Numerical
Degrees (0–180)
angle(c 2345,c 6789,c 123)=97
compass
2 cities
Categorical
8 directions
compass(c 1234,c 5678)=NE
inside
1 + n cities
Categorical


### TRUE/FALSE


inside(c 9012;c 3456,...)=FALSE
perimeter
n cities
Numerical
Scaled coords
perimeter(c 4567,c 8901,...)=2856
crossing
4 cities
Categorical


### TRUE/FALSE


cross(c 2345,c 6789;c 123,c 4567)=TRUE


Table 1: Summary of 7 geometric tasks. Numerical outputs are integers; “scaled coords” refers to
the ×10 coordinate system (Sec. C.1). Categorical tasks have discrete outputs: compass uses 8
cardinal directions (N, NE, E, SE, S, SW, W, NW), while inside and crossing are binary. The
inside task tests if the first city lies within the convex hull of the remaining cities; crossing
tests if line segment (c1, c2) intersects segment (c3, c4).
It is important to note that for all tasks we study, queries that don’t explicitly involve Atlantis
cities maintain identical outputs after Atlantis is introduced—ensuring we can cleanly measure
integration of new knowledge. While our framework could be extended to study tasks where existing answers change (e.g., counting cities within a radius would yield different results after adding
Atlantis), enabling investigation of phenomena like the reversal curse (Berglund et al., 2024),
we focus here on the simpler case of integrating new entities while preserving existing knowledge.
Dataset Sizes
Each pretraining set consists of 1M rows of data per task. For fine-tuning, the
dataset consists of: (1) 100k rows of the target task containing at least one Atlantis city, (2)
20k rows randomly sampled from the original pretraining data to prevent catastrophic forgetting,
and (3) 256 rows per task (without Atlantis) to elicit multi-task performance. For the baseline
experiment where Atlantis is included during pretraining (green line in Fig. 6d), we use 1M
rows per task but sample cities uniformly without treating Atlantis specially.
C.3


### MODEL AND TRAINING


Tokenization
We use character-level tokenization with 98 ASCII tokens (excluding space, which
serves as the delimiter), plus special tokens for beginning-of-sequence (BOS), end-of-sequence
(EOS), and padding (PAD). Each task query and answer is tokenized character-by-character
(e.g., dist(c 0865,c 4879)=769 becomes d i s t ( c
0 8 6 5 , c
4 8 7
9 ) = 7 6 9).
This character-level scheme is intentional. While assigning each city and task a dedicated token
would simplify learning, such synthetic-friendly tokenization does not reflect how real language
models operate. LLMs must handle multi-token entities, variable-length prompts (our task prefixes
have different lengths), computations at different sequence positions, and irregularly tokenized con18


![Table2-1: This image shows a scientific table detailing the hyperparameters used for training a machine learning model. The table is organized into two columns: "Hyperparameter" on the left and "Value" on the right. The table contains nine rows of data: 1. \*\*Optimizer:\*\* The value is "AdamW," followed by a citation in parentheses: "(Loshchilov & Hutter, 2019)". In the image, the citation text is highlighted with two green rectangular boxes. 2. \*\*Learning rate:\*\* The value is set to $3 \times 10^{-4}$ (0.0003). 3. \*\*Weight decay:\*\* The value is 0.01. 4. \*\*Scheduler:\*\* The value is "Linear with warmup," indicating a learning rate schedule that increases linearly during the initial phase and then decreases. 5. \*\*Warmup steps:\*\* The value is 50, specifying the number of initial steps for the warmup phase of the scheduler. 6. \*\*Batch size:\*\* The value is 128. 7. \*\*Max sequence length:\*\* The value is 256, likely referring to the number of tokens or data points in an input sequence. 8. \*\*Total training rows:\*\* The value is 42M (42 million), indicating the scale of the dataset used for training. 9. \*\*Initialization scale:\*\* The value is 0.1 (std), suggesting that model weights were initialized using a distribution with a standard deviation of 0.1. The table is formatted with horizontal lines at the top and bottom of the header row and a single horizontal line at the very bottom of the table. The text is in a serif font, and the headers are bolded. The overall configuration describes a standard setup for training modern transformer-based or deep learning models on a large-scale dataset.](figures/Table2-1.png)
*Table 2: Pretraining hyperparameters.*

Preprint
tent (e.g., numbers in LaTeX). Preliminary experiments exploring pitfalls of next-token prediction
(Bachmann & Nagarajan, 2025) showed that tokenization details qualitatively affect results. We
therefore chose character-level tokenization to better approximate realistic sequence modeling conditions.
City ID Assignment
City IDs are randomly assigned from the range [0, 9999], ensuring no geographic information leaks through the identifier. This random assignment means the model cannot
exploit ID patterns to infer coordinates.
Architecture
We use the Qwen2 (Yang et al., 2024) decoder-only transformer architecture with
hidden size 128, 4 attention heads, and 6 layers.
Pretraining
We train models autoregressively on the full sequence (no prompt masking). While
we observed training speedup when masking loss computation on the prompt side, we deliberately
avoid this optimization to maintain similarity with standard autoregressive language model pretraining. All pretraining runs see 42M rows regardless of dataset size (e.g., 42 epochs for 1M rows, 6
epochs for 7M rows). Table 2 summarizes the hyperparameters.
Hyperparameter
Value
Optimizer
AdamW (Loshchilov & Hutter, 2019)
Learning rate
3 × 10−4
Weight decay
0.01
Scheduler
Linear with warmup
Warmup steps
Batch size
Max sequence length
Total training rows
42M
Initialization scale
0.1 (std)


Table 2: Pretraining hyperparameters.
Fine-Tuning
Fine-tuning starts from the final pretrained checkpoint. We use a reduced learning
rate of 1 × 10−5 (30× smaller than pretraining) to avoid catastrophic forgetting. The fine-tuning
dataset consists of 100k rows per task containing at least one Atlantis city. We train for 30
epochs with batch size 128. We observed significant degradation in performance for both the finetuned task and original (non-Atlantis) tasks when using a larger batch size of 512. All other
hyperparameters (optimizer, weight decay, scheduler, warmup) remain the same as pretraining.
D


### ANALYSIS METHODS


D.1


### EVALUATION


Generation Protocol
For evaluation, we use teacher forcing up to the “=” sign (the prompt), then
generate autoregressively at temperature zero until reaching the EOS token or a maximum of 128
tokens (sufficient for all tasks). All trained models achieve perfect parse accuracy—outputs always
match the expected format (integers for numerical tasks, valid categories for categorical tasks).
Task-Specific Metrics
Categorical tasks (compass, inside, crossing) are evaluated using
accuracy. Numerical tasks are evaluated using absolute error: distance (scaled coordinate units),
triarea (scaled coordinate units2), angle (degrees), and perimeter (scaled coordinate units).
Normalized Improvement
To compare generalization across tasks with different metrics and
scales, we define a normalized improvement score that maps performance to [0, 1], where 0 indicates no improvement over the Atlantis baseline (before fine-tuning) and 1 indicates matching
the pretrained model’s performance on standard cities.

Preprint
For error-based tasks (distance, triarea, angle, perimeter), where lower is better:


### NI =


log(baselineatlantis/error)
log(baselineatlantis/baselinestandard)
(2)
The logarithmic scaling ensures multiplicative improvements are treated equally (e.g., reducing error
from 1000 to 100 is weighted the same as 100 to 10).
For accuracy-based tasks (compass, inside, crossing), where higher is better:


### NI =


accuracy −baselineatlantis
baselinestandard −baselineatlantis
(3)
Note that normalized improvement can slightly exceed 1.0 if, by chance, Atlantis cities perform
better than the average pretrained city on some task.
D.2


### REPRESENTATION EXTRACTION


We extract representations from the residual stream after transformer blocks, specifically at layers
3, 4, 5, and 6 of our 6-layer model. Unless otherwise specified, all representation analyses in this
paper use layer 5 representations.
To extract city representations, we pass a task prefix followed by a city ID through the model. For
single-task models, we use the corresponding task prefix. For multi-task models (2-task and 3-task),
we use the first task in the combination as the prefix. We verified that the choice of task prefix has
negligible effect on the extracted city representations.
For a city with ID 1234, the input sequence is:
<bos> d i s t ( c
1 2 3
,
We extract and concatenate the representations of two tokens: (1) the last digit of the city ID and
(2) the following delimiter token (typically a comma). This yields a 256-dimensional representation (128 × 2) per city, which we use for both PCA visualization and linear probing.
Omitting cities with leading zeros
We omit cities with IDs starting with 0, 00, or 000 from
representation analyses. These cities form distinct clusters in representation space, separate from
cities with IDs starting with non-zero digits. We hypothesize this occurs because the digit 0 has
special semantic status: in numerical outputs (distances, angles, areas), leading zeros never appear
(e.g., “=769” not “=0769”), so the model learns to treat 0 differently when it appears as a leading
digit. When 0 appears at the start of a city ID, the model may encode a feature indicating “this is an
identifier, not a number,” causing these cities to cluster separately. To ensure consistent evaluation
across all cities, we exclude IDs matching the pattern ˆ[0][0-9]*$ (i.e., any ID starting with
zero).
D.3


### LINEAR PROBING & PCA


We use the representations described in Sec. D.2 for both PCA visualization and linear probing.
Linear Probing
We train linear probes to predict city coordinates (x, y) from the 256-dimensional
representations. We use a train/test split of 3250/1250 cities, training separate probes for x and y
coordinates via ordinary least squares (OLS) without regularization. We report R2 scores and mean
absolute error in scaled coordinate units.
PCA
For visualization, we apply PCA to the representations and plot the first two or three principal components. We use consistent color coding based on geographic region to enable visual
comparison across models and seeds.

Preprint
Reconstruction Error
To quantify how well new entities (Atlantis cities) are integrated into
the learned manifold, we train linear probes exclusively on non-Atlantis cities and evaluate
reconstruction error on held-out Atlantis representations. Reconstruction error is measured as
the absolute Euclidean distance between predicted and true coordinates. Large reconstruction errors
indicate that new entities are encoded in different subspaces than the original cities.
D.4


### CENTERED KERNEL ALIGNMENT


We use Centered Kernel Alignment (CKA) (Kornblith et al., 2019) to measure representational
similarity between models. Given two representation matrices X ∈Rn×d1 and Y ∈Rn×d2 (same
n cities, potentially different dimensions), we compute linear kernel matrices K = XXT and L =
Y Y T , center them, and compute:


### CKA(X, Y ) =



### ⟨K, L⟩F



### ∥K∥F ∥L∥F


(4)
where ⟨·, ·⟩F denotes the Frobenius inner product. CKA yields a similarity score in [0, 1] that is
invariant to orthogonal transformations and isotropic scaling.
For each pair of models, we extract city representations (Sec. D.2) and compute CKA between the
resulting matrices. We filter cities to exclude Atlantis and IDs starting with zeros. We report
CKA values at layers 3, 4, 5, and 6, with layer 5 as the default unless otherwise specified.
E


### ADDITIONAL EXPERIMENTS & RESULTS


E.1


### TRAINING DYNAMICS


Fig. 8 shows training dynamics for all seven single-task models. Each panel displays three rows of

![Figure8-1: This figure consists of seven panels (a through g), each representing a different geometric task used to train a neural network. The tasks are: a) Distance, b) Triangle Area, c) Angle, d) Compass, e) Inside, f) Perimeter, and g) Crossing. Each panel contains three vertically stacked line plots sharing a common x-axis: "Gradient Steps" on a logarithmic scale ranging from approximately $10^2$ to $5 \times 10^5$. ### Common Layout and Legend Across all panels, the three sub-plots follow a consistent format: 1. \*\*Top Plot (Loss):\*\* Displays "Training Loss" (blue line) and "Validation Loss" (orange line) in nats. 2. \*\*Middle Plot (Performance Metrics):\*\* Contains two y-axes. The left y-axis (green) shows a task-specific error or accuracy metric (green line). The right y-axis (red) shows "Coordinate $R^2$" (red line), measuring how well the model's internal representations can linearly reconstruct the input coordinates. 3. \*\*Bottom Plot (Linear Probing):\*\* Displays the "Linear Probing Distance Error" (magenta line) on a logarithmic y-axis, representing the error when trying to decode Euclidean distance from the model's hidden layers using a linear probe. --- ### Detailed Panel Descriptions \*\*a) Distance:\*\* \* \*\*Loss:\*\* Both losses remain flat until $\sim10^4$ steps, then drop steadily. Final training loss is $\sim0.75$, validation $\sim0.80$. \* \*\*Metrics:\*\* "Distance Error" (green) starts high ($\sim1000$) and drops sharply after $10^4$ steps to $\sim3$. Simultaneously, "Coordinate $R^2$" (red) rises from $0.0$ to nearly $1.0$. \* \*\*Probing:\*\* Error remains at $500$ until $10^4$ steps, then drops to $\sim100$. \*\*b) Triangle Area:\*\* \* \*\*Loss:\*\* Shows a two-stage drop. A small drop at $5 \times 10^3$ and a larger drop after $3 \times 10^4$. \* \*\*Metrics:\*\* "Area Error" (green) stays at $100,000$ until $3 \times 10^4$ steps, then falls to $\sim1,000$. Coordinate $R^2$ (red) mirrors this, rising to $1.0$ during the second drop. \* \*\*Probing:\*\* Error drops significantly after $3 \times 10^4$ steps, reaching $\sim80$. \*\*c) Angle:\*\* \* \*\*Loss:\*\* Similar two-stage decline as Triangle Area. \* \*\*Metrics:\*\* "Angle Error (deg)" (green) remains at $80^\circ$ until $4 \times 10^4$ steps, then drops sharply to $\sim2^\circ$. Coordinate $R^2$ (red) reaches $1.0$ at the same time. \* \*\*Probing:\*\* Shows a very sharp drop at $4 \times 10^4$ steps, falling from $500$ to $\sim60$. \*\*d) Compass (Direction):\*\* \* \*\*Loss:\*\* Drops early ($2 \times 10^3$ steps) and plateaus around $0.65$. \* \*\*Metrics:\*\* "Compass Accuracy" (green) rises from $0.25$ to $\sim0.90$ between $10^4$ and $10^5$ steps. Interestingly, Coordinate $R^2$ (red) only reaches $\sim0.6$, suggesting coordinates are not fully recovered. \* \*\*Probing:\*\* The error remains high and flat ($>400$) throughout training. \*\*e) Inside (Point-in-polygon):\*\* \* \*\*Loss:\*\* Drops at $2 \times 10^3$ and plateaus. \* \*\*Metrics:\*\* "Inside Accuracy" (green) rises to $\sim0.95$. Coordinate $R^2$ (red) rises sharply but late (at $10^5$ steps) to $\sim0.8$. \* \*\*Probing:\*\* Error stays at $500$ until $10^5$ steps, then drops to $\sim150$. \*\*f) Perimeter:\*\* \* \*\*Loss:\*\* Gradual, continuous decline after $10^4$ steps. \* \*\*Metrics:\*\* "Perimeter Error" (green) drops from $1000$ to $\sim20$. Coordinate $R^2$ (red) reaches $1.0$ by $10^5$ steps. \* \*\*Probing:\*\* Error drops steadily after $2 \times 10^4$ steps to $\sim50$. \*\*g) Crossing (Line intersection):\*\* \* \*\*Loss:\*\* Drops early and stays low, with some noise/spikes in training loss. \* \*\*Metrics:\*\* "Crossing Accuracy" (green) rises to nearly $1.0$. However, Coordinate $R^2$ (red) remains very low ($<0.2$). \* \*\*Probing:\*\* Error remains completely flat at $500$, indicating no distance information is captured. ### Key Insights The figure demonstrates that for many geometric tasks (Distance, Area, Angle, Perimeter), the model undergoes a "phase change" where it suddenly learns to represent underlying coordinates (high $R^2$) and solve the task. However, for qualitative tasks like "Compass" or "Crossing," the model can achieve high task accuracy without necessarily learning a high-fidelity coordinate map or Euclidean distance relationships, as evidenced by the low $R^2$ and high probing errors in those panels.](figures/Figure8-1.png)
*Figure 8: Training dynamics for all single-task models. (a) distance, (b) trianglearea, (c) angle, (d) compass, (e) inside, (f) perimeter, (g) crossing. Each panel shows three rows: (top) training loss (blue) and validation loss (orange); (middle) task-specific metric (green, left axis) and linear probe coordinate R2 (red, right axis); (bottom) linear probing distance error (magenta). All plots use log-scale x-axis for gradient steps.*

metrics over gradient steps: (top) training and validation loss, (middle) task-specific performance
metric alongside linear probe R2 for coordinate decoding, and (bottom) linear probing distance error
measuring how accurately city coordinates can be reconstructed from representations.
Several patterns emerge across tasks. First, all tasks except crossing eventually achieve high
coordinate R2 (red curves reaching ∼1.0), indicating that world representations form reliably across
diverse geometric objectives. Second, the relationship between loss, task performance, and coordinate decodability varies across tasks. Third, crossing (panel g) fails entirely in single-task
training. Loss remains high, accuracy stays near chance, and coordinate R2 never rises, consistent
with the main text observation that this task requires multi-task scaffolding.
Representation Dynamics.
Fig. 9 visualizes how internal representations evolve during training

![Figure9-1: This figure presents a grid of 18 three-dimensional scatter plots organized into three rows and six columns. The visualization tracks the evolution of data representations in a neural network or machine learning model over time, using Principal Component Analysis (PCA) to project high-dimensional data into a 3D space defined by PC1, PC2, and PC3. ### Layout and Organization - \*\*Columns (Time/Training Steps):\*\* The columns represent sequential stages of training or processing, labeled with increasing numerical values at the top: 8204, 24612, 49224, 123060, 188692, and 328146. - \*\*Rows (Metric/Feature Type):\*\* The rows are labeled on the left as "Distance," "Angle," and "Compass." These likely represent different types of features or loss functions being analyzed. - \*\*Axes:\*\* Each individual plot features three axes: PC1 (horizontal depth), PC2 (horizontal width), and PC3 (vertical height). The scale of these axes increases significantly from left to right as the training progresses. ### Row-by-Row Analysis #### 1. Distance Row - \*\*Initial State (8204):\*\* The data points are tightly packed in a single, undifferentiated globular cluster centered around the origin. Colors (representing different classes or categories) are completely intermixed. - \*\*Intermediate Stages (24612 - 49224):\*\* The single cluster begins to elongate and fragment. Distinct "arms" or branches start to emerge, and colors begin to group together. - \*\*Final Stages (123060 - 328146):\*\* The data has organized into a highly structured, "C-shaped" or semi-circular manifold. Several distinct, dense clusters of specific colors (e.g., blue, red, purple, orange) have separated from the main body, indicating the model has learned to distinguish between different categories based on distance metrics. #### 2. Angle Row - \*\*Initial State (8204):\*\* Similar to the Distance row, the data starts as a chaotic, multi-colored sphere. - \*\*Intermediate Stages (24612 - 49224):\*\* The data transitions into a more linear or "comet-like" structure. A dense head of mixed colors is visible, with a tail of more separated colors extending outward. - \*\*Final Stages (123060 - 328146):\*\* The representation matures into a distinct "U-shaped" or "horseshoe" manifold. While there is clear separation of color clusters (especially orange, red, and purple), the overall structure is more continuous and curved compared to the Distance row. #### 3. Compass Row - \*\*Initial State (8204):\*\* Starts as a centralized, mixed cluster. - \*\*Intermediate Stages (24612 - 49224):\*\* This row shows the fastest separation of clusters. By step 24612, distinct "islands" of color (orange, red, green, light blue) are already visible and physically separated in the PCA space. - \*\*Final Stages (123060 - 328146):\*\* The data settles into a stable configuration of several isolated, tight clusters. Unlike the "Distance" and "Angle" rows which form continuous manifolds (lines or curves), the "Compass" row results in discrete, well-separated groupings, suggesting a highly categorical representation. ### Key Scientific Insights - \*\*Representation Learning:\*\* The figure illustrates the process of "manifold learning," where a model takes disorganized input and organizes it into a structured geometry where similar items are grouped together. - \*\*Convergence:\*\* In all three rows, the change between the last two columns (188692 and 328146) is minimal, suggesting the model's representations have reached a steady state or converged. - \*\*Metric Differences:\*\* The different rows show that the choice of feature (Distance vs. Angle vs. Compass) significantly impacts the final geometry of the latent space. "Compass" features appear to produce the most discrete and separated class representations, while "Distance" and "Angle" produce more continuous, interconnected structures.](figures/Figure9-1.png)
*Figure 9: Representation dynamics during training. Rows: distance (top), angle (middle), compass (bottom). Columns show PCA projections at gradient steps 8204, 24612, 49224, 123060, 188692, and 328146 (left to right). Cities are colored by geographic region.*

via PCA projections at six checkpoints. A striking pattern emerges: once a representational structure
forms, it remains largely fixed throughout the subsequent training phase where task accuracy continues to improve. Examining the gradient steps, representations are essentially fixed in the first ∼15%
of training, remaining static while loss slowly decreases and accuracy rises. The distance task
(top row) establishes its thread-like structure early; angle (middle row) settles into a 2D manifold;
compass (bottom row) forms fragmented regional clusters, all within the first few checkpoints,
with minimal subsequent change. What determines when representations stop evolving remains unclear, though it appears correlated with the initial loss drop. This may relate to recently observed
gradient dynamics in language model training, where loss deceleration phases exhibit qualitatively
different learning behavior (Mircea et al., 2025).

Preprint


Figure 8: Training dynamics for all single-task models. (a) distance, (b) trianglearea,
(c) angle, (d) compass, (e) inside, (f) perimeter, (g) crossing. Each panel shows three
rows: (top) training loss (blue) and validation loss (orange); (middle) task-specific metric (green,
left axis) and linear probe coordinate R2 (red, right axis); (bottom) linear probing distance error
(magenta). All plots use log-scale x-axis for gradient steps.


Figure 9: Representation dynamics during training. Rows: distance (top), angle (middle),
compass (bottom). Columns show PCA projections at gradient steps 8204, 24612, 49224, 123060,
188692, and 328146 (left to right). Cities are colored by geographic region.


![Figure10-1: Figure 10-1: This figure presents a grid of 18 three-dimensional scatter plots, organized into six columns and three rows. The figure visualizes the latent representations of neural network activations using Principal Component Analysis (PCA). Each column corresponds to a specific geometric or relational task, while the three rows likely represent different experimental conditions, time points, or model layers. ### Layout and Axes The figure is organized into six columns, each labeled at the top: 1. \*\*Distance\*\* 2. \*\*Triangle Area\*\* 3. \*\*Angle\*\* 4. \*\*Compass\*\* 5. \*\*Inside\*\* 6. \*\*Perimeter\*\* Each individual plot is a 3D scatter plot with axes labeled \*\*PC1\*\*, \*\*PC2\*\*, and \*\*PC3\*\*, representing the first three principal components of the data. The axes scales vary between plots, reflecting the variance captured in different tasks and conditions. The data points in every plot are color-coded, where different colors represent distinct categories, classes, or ranges of values relevant to the specific task. ### Column-by-Column Analysis #### 1. Distance \* \*\*Top Row:\*\* Shows a complex, branching structure. Points are clustered into distinct "arms" or lobes. A large yellow cluster is prominent on the left, while red, blue, and cyan clusters form a more central, interconnected structure. \* \*\*Middle Row:\*\* The data forms a more elongated, "V-shaped" manifold. The yellow points form one distinct branch, while purple and green points are concentrated near the vertex. \* \*\*Bottom Row:\*\* Displays a similar V-shaped or "boomerang" structure, but with a more dense concentration of points at the junction. Red and purple points are clustered at one end of the manifold. #### 2. Triangle Area \* \*\*Top Row:\*\* The points form a hollow, ring-like or "torus" structure. Colors (orange, green, purple, cyan) are distributed around the perimeter of this shape, suggesting a cyclical or continuous relationship in the data. \* \*\*Middle Row:\*\* A more globular, dense cluster with some internal structure. The colors are somewhat mixed but show local grouping, with cyan and purple points forming a central core. \* \*\*Bottom Row:\*\* Shows a very clear circular or "O-shaped" manifold. The colors are arranged sequentially around the ring (red to orange to purple to green to blue), indicating that the latent space has mapped the "Triangle Area" feature onto a periodic or circular dimension. #### 3. Angle \* \*\*Top Row:\*\* The data forms a curved, "C-shaped" manifold. Yellow points are isolated at one tip, while other colors (green, purple, cyan, red) follow the curve of the manifold. \* \*\*Middle Row:\*\* A dense, vertical "tower" or "column" structure. The points are layered by color, with orange at the bottom, followed by green, cyan, and red at the top. \* \*\*Bottom Row:\*\* Similar to the top row, it shows a curved, semi-circular manifold. The points are well-separated by color along the trajectory of the curve. #### 4. Compass \* \*\*Top Row:\*\* Points are organized into several distinct, localized clusters. There is a clear separation between the orange, red, green, and blue groups, suggesting the model has learned discrete directional categories. \* \*\*Middle Row:\*\* Shows a more integrated, "starburst" or "cross" shape. The clusters are still distinct but appear to radiate from a common center. \* \*\*Bottom Row:\*\* Displays a series of well-defined, isolated clusters arranged in a roughly circular pattern in 3D space. Each color (orange, red, purple, green, cyan) occupies a specific "island" in the latent space. #### 5. Inside \* \*\*Top Row:\*\* A dense, spherical cloud of points. The colors are highly interleaved, though there is a slight gradient from purple/cyan on the left to orange/red on the right. \* \*\*Middle Row:\*\* Similar to the top row, a dense globular cluster. The distribution of colors appears more stratified here, with distinct layers of pink, purple, and orange. \* \*\*Bottom Row:\*\* A flattened, "pancake" or disc-like distribution. The colors are arranged in concentric or adjacent zones, with red/orange on one side and green/blue on the other. #### 6. Perimeter \* \*\*Top Row:\*\* A complex, "hook-shaped" manifold. The data follows a clear path, starting with yellow points at one end, curving through orange and red, and ending in a dense blue/purple cluster. \* \*\*Middle Row:\*\* A very smooth, "arch" or "rainbow" shaped manifold. The colors transition linearly along the length of the arch (green to blue to red to orange). \* \*\*Bottom Row:\*\* Shows two distinct, curved trajectories or "ribbons." One ribbon is dominated by blue and purple points, while the other contains green, cyan, and orange points. ### Key Scientific Insights \* \*\*Manifold Structure:\*\* The figure demonstrates that the neural network represents different geometric concepts as specific topological manifolds (rings, arches, clusters, or V-shapes). \* \*\*Task-Specific Encoding:\*\* Tasks like "Triangle Area" and "Perimeter" result in continuous, smooth manifolds (rings and arches), suggesting the model treats these as continuous variables. In contrast, "Compass" results in discrete clusters, suggesting a categorical representation of direction. \* \*\*Dimensionality Reduction:\*\* PCA effectively reveals that despite the high dimensionality of neural activations, the "knowledge" of these geometric tasks is often compressed into low-dimensional (1D or 2D) structures embedded within the 3D space. \* \*\*Consistency Across Rows:\*\* While the exact orientation and shape of the manifolds change across the rows, the fundamental topology (e.g., the ring for Triangle Area or the arch for Perimeter) remains relatively consistent, indicating robust feature representation.](figures/Figure10-1.png)
*Figure 10: Representation visualizations for single-task models across multiple seeds. Each column shows a different task; each row shows a different random seed. Cities are colored by geographic region. Despite seed variability, task-specific geometric patterns are visible.*

Preprint
E.2


### QUALITATIVE REPRESENTATIONS


Fig. 10 shows PCA projections of city representations for single-task models across three random seeds (rows). The distance task consistently produces characteristic thread-like structures.
Angle and perimeter often form larger 2D manifold-like structures. triangle area tends
to produce arc-shaped geometries. Compass forms local clusters corresponding to directional categories, while inside produces a more global, diffuse structure.
While there is some seed-to-seed variability within each task, the broader categories remain distinguishable: distance representations are qualitatively distinct from the cluster-based representations of compass and inside, and both differ from the manifold-like structures produced by
triangle area, angle, and perimeter.


Figure 10: Representation visualizations for single-task models across multiple seeds. Each
column shows a different task; each row shows a different random seed. Cities are colored by
geographic region. Despite seed variability, task-specific geometric patterns are visible.


![Figure11-1: Figure 11-1: This figure consists of four heatmaps arranged in a 2x2 grid, representing the Centered Kernel Alignment (CKA) similarity between different task representations across four specific layers of a neural network: Layer 3, Layer 4, Layer 5, and Layer 6. ### General Structure and Legend Each heatmap is a 7x7 matrix. The rows and columns are labeled with abbreviations representing seven different tasks: \* \*\*D\*\*: Depth estimation \* \*\*T\*\*: Texture classification \* \*\*A\*\*: Surface normals (Aesthetics/Geometry) \* \*\*Co\*\*: Content/Colorization \* \*\*I\*\*: Inpainting \* \*\*P\*\*: Parts segmentation \* \*\*Cr\*\*: Curvature A vertical color bar on the right side of the figure provides the scale for the CKA values. The scale ranges from \*\*0.0 (black/dark purple)\*\*, indicating no similarity, to \*\*1.0 (light yellow/white)\*\*, indicating identical representations. Intermediate values are represented by shades of purple, pink, and orange. Each cell in the matrices contains a numerical value (the mean CKA score) and a ± value (the standard deviation). ### Panel-by-Panel Description #### Layer 3 (Top Left) This layer shows relatively low similarity across most tasks. \* \*\*Diagonal Values:\*\* The self-similarity (e.g., D-D, T-T) ranges from 0.22 to 0.82. Notably, task \*\*P\*\* (0.82) has the highest self-consistency, while \*\*Co\*\* (0.22) and \*\*Cr\*\* (0.23) are much lower. \* \*\*Cross-Task Similarity:\*\* Most off-diagonal values are low, often below 0.30. The highest cross-task similarities are between \*\*A and P\*\* (0.55) and \*\*T and P\*\* (0.38). \* \*\*Task Cr:\*\* This task shows almost zero similarity (0.01 to 0.05) with all other tasks except itself. #### Layer 4 (Top Right) Representational similarity increases significantly in this layer compared to Layer 3. \* \*\*Diagonal Values:\*\* Self-similarity is much higher, with \*\*T\*\* (0.82), \*\*A\*\* (0.92), and \*\*P\*\* (0.93) showing very high internal consistency. \* \*\*Task Clusters:\*\* A strong cluster of similarity emerges between \*\*T, A, I, and P\*\*. For example, the similarity between \*\*T and A\*\* is 0.76, and between \*\*A and P\*\* is 0.84. \* \*\*Task Cr:\*\* Remains highly isolated, with values of 0.00 to 0.05 when compared to other tasks. #### Layer 5 (Bottom Left) The similarity between tasks continues to deepen and broaden. \* \*\*Diagonal Values:\*\* Most tasks (except Cr) have self-similarity scores above 0.80, with \*\*A\*\* and \*\*P\*\* reaching 0.93. \* \*\*Broad Similarity:\*\* High similarity scores (0.60 to 0.88) are now common across \*\*D, T, A, Co, I, and P\*\*. For instance, \*\*Co\*\* now shows high similarity with \*\*T\*\* (0.77), \*\*A\*\* (0.79), and \*\*P\*\* (0.86). \* \*\*Task Cr:\*\* Remains the outlier, with near-zero similarity to all other tasks and a low self-similarity of 0.12. #### Layer 6 (Bottom Right) The patterns in Layer 6 are similar to Layer 5 but show a slight divergence in some tasks. \* \*\*Strongest Links:\*\* The highest similarities remain between \*\*T, A, Co, and P\*\*, with many values exceeding 0.80. \* \*\*Task I (Inpainting):\*\* Shows a decrease in similarity to other tasks compared to Layer 5 (e.g., I-D drops from 0.48 to 0.39). \* \*\*Task Cr:\*\* Its isolation is most extreme here, with 0.00 similarity to almost every other task and its self-similarity dropping to a very low 0.03. ### Key Insights 1. \*\*Representational Convergence:\*\* As data moves from Layer 3 to Layer 5, the representations for most tasks (D, T, A, Co, I, P) become increasingly similar to one another, suggesting the network develops a shared feature space for these objectives. 2. \*\*Task Clusters:\*\* Tasks related to geometry and segmentation (\*\*A\*\* and \*\*P\*\*) and texture (\*\*T\*\*) show the highest and earliest similarity. 3. \*\*The Outlier (Cr):\*\* The Curvature task (\*\*Cr\*\*) consistently fails to align with any other task representations across all layers, indicating it requires a fundamentally different feature set than the other six tasks. 4. \*\*Layer Progression:\*\* Layer 5 appears to be the point of maximum representational overlap for the majority of tasks before slight specialization begins to reappear in Layer 6.](figures/Figure11-1.png)
*Figure 11: CKA matrices for single-task models across layers. Each cell shows mean ± SEM across 3 seeds. D=distance, T=triangle area, A=angle, Co=compass, I=inside, P=perimeter, Cr=crossing. CKA increases in later layers; distance shows consistently lower cross-task similarity.*

Preprint
E.3


### ADDITIONAL CKA RESULTS


Single-Task CKA Across Layers.
Fig. 11 shows CKA matrices for single-task models at layers
3, 4, 5, and 6. Each cell shows mean ± SEM across 3 seeds. We observe: (1) CKA values increase
from layer 3 to layers 4–6, indicating that world representations become more consistent in later layers; (2) the distance task (D) shows lower CKA with other tasks across all layers, consistent with
its divergent representational geometry; (3) crossing (Cr) shows near-zero CKA due to training
failure in single-task settings; (4) diagonal entries (same task) can show significant variability, indicating that even identical training objectives can yield different representational solutions.


Figure 11: CKA matrices for single-task models across layers.
Each cell shows mean ±
SEM across 3 seeds. D=distance, T=triangle area, A=angle, Co=compass, I=inside, P=perimeter,
Cr=crossing. CKA increases in later layers; distance shows consistently lower cross-task similarity.
Two-Task CKA.
Fig. 12 shows the CKA matrix for two-task models at layer 5. Compared to

![Figure12-1: Figure 12-1: This figure presents a heatmap representing a similarity matrix between seven different categories or models, labeled with the abbreviations D, T, A, Co, I, P, and Cr. The similarity is measured using Centered Kernel Alignment (CKA), a metric used to compare internal representations in neural networks or datasets. ### Layout and Axes The figure is a square 7x7 grid. Both the horizontal (top) and vertical (left) axes are labeled with the same set of seven identifiers: \* \*\*D\*\*: Likely representing "Detection" or a specific dataset/model starting with D. \* \*\*T\*\*: Likely representing "Tracking." \* \*\*A\*\*: Likely representing "Action." \* \*\*Co\*\*: Likely representing "Counting." \* \*\*I\*\*: Likely representing "Identification." \* \*\*P\*\*: Likely representing "Pose." \* \*\*Cr\*\*: Likely representing "Crowd." ### Data and Color Scale \* \*\*Color Gradient\*\*: To the right of the grid is a vertical color bar labeled "CKA." The scale ranges from 0.0 (black/dark purple) to 1.0 (light cream/yellow). The intermediate colors transition through dark purple, magenta, and orange-red. \* \*\*Cell Values\*\*: Each cell in the matrix contains a numerical value representing the mean CKA score, followed by a standard deviation (e.g., $0.95 \pm 0.013$). \* \*\*Heatmap Trends\*\*: All values in the matrix are high, ranging from a minimum of 0.84 to a maximum of 0.97. This indicates a very high degree of representational similarity across all compared categories. \* \*\*Diagonal Elements\*\*: The diagonal from the top-left to the bottom-right represents the self-similarity of each category. These values are the highest in the matrix, ranging from 0.89 (for Co) to 0.97 (for Cr), appearing in the lightest cream color. \* \*\*Off-Diagonal Elements\*\*: These represent cross-category similarities. Notable high similarities include T and Cr (0.93), T and P (0.93), and I and P (0.92). The lowest similarities (0.84) occur between D and I, T and Co, and Co and I. ### Annotations \* \*\*Red Triangles\*\*: Several cells in the upper and lower triangles of the matrix feature a small red triangle in their top-right corner. These appear to mark specific pairs of interest or significant correlations, though their specific meaning is not defined in the legend. They are present in cells such as (D, Co), (D, I), (T, I), (T, P), (A, P), (A, Cr), (Co, D), (Co, Cr), (I, D), (I, T), (P, T), (P, A), (Cr, A), and (Cr, Co). ### Key Insights The figure demonstrates that the representations for these seven tasks (D, T, A, Co, I, P, Cr) are highly consistent and similar to one another, with all CKA scores exceeding 0.84. This suggests a strong shared underlying feature space between these different domains or tasks. The highest cross-task similarities involve Tracking (T), Pose (P), and Crowd (Cr), while Counting (Co) shows slightly lower self-similarity and cross-similarity compared to the others.](figures/Figure12-1.png)
*Figure 12: CKA matrix for two-task models at layer 5. Mean ± SEM across 3 seeds. All pairs show high alignment (>0.84), substantially higher than single-task models.*

single-task models (Fig. 11, layer 5), two-task training substantially increases representational alignment: all off-diagonal entries exceed 0.84, compared to values as low as 0.48 for single-task models.
Notably, diagonal entries (same task combination, different seeds) show minimum CKA of 0.89, indicating that multi-task training also reduces inter-seed variance. For diagonal entries, we exclude
same-seed comparisons (which trivially yield 1.0) and report only the upper triangle since the matrix is symmetric. This confirms the main text finding that multi-task training drives representational
convergence.
CKA vs. Task Count (Per-Seed).
Fig. 13 shows the same CKA vs. task count analysis as Fig. 3(d)

![Figure13-1: Figure 13-1: This figure consists of three side-by-side line graphs, labeled "Seed 1," "Seed 2," and "Seed 3," which illustrate the Centered Kernel Alignment (CKA) similarity scores across different layers of a neural network as the number of tasks increases. ### General Layout and Axes Each of the three panels shares the same structure: - \*\*Y-axis:\*\* Represents "CKA" (Centered Kernel Alignment), a measure of similarity between neural representations, ranging from 0.0 to 1.0. - \*\*X-axis:\*\* Represents the number of tasks trained, with categorical markers for "1 Task," "2 Tasks," and "3 Tasks." - \*\*Data Points:\*\* Individual data points are shown as small, semi-transparent dots (jittered for visibility), while the mean for each layer is represented by a solid square connected by lines. - \*\*Error Bars:\*\* Vertical bars extend from each mean square, representing the standard error or confidence interval. - \*\*Legend:\*\* Located at the bottom of the figure, indicating four layers color-coded as follows: - \*\*Layer 3:\*\* Blue - \*\*Layer 4:\*\* Orange - \*\*Layer 5:\*\* Green - \*\*Layer 6:\*\* Red ### Data Trends and Insights Across all three seeds, several consistent trends are observed: 1. \*\*Layer Hierarchy and Similarity:\*\* - \*\*Layer 3 (Blue)\*\* consistently shows the lowest CKA scores, starting between 0.2 and 0.3 for 1 Task and rising slightly to approximately 0.4 by 3 Tasks. This suggests that the representations in this earlier layer are less similar across different task conditions compared to deeper layers. - \*\*Layers 4, 5, and 6\*\* exhibit significantly higher CKA scores, generally starting above 0.5 and often reaching between 0.8 and 0.9. This indicates that deeper layers develop more stable or shared representations as more tasks are added. 2. \*\*Effect of Increasing Tasks:\*\* - There is a general upward trend in CKA similarity for all layers as the number of tasks increases from 1 to 2. - Between 2 Tasks and 3 Tasks, the similarity scores tend to plateau or show a slight decrease (most notably in Seed 1 for Layer 5). - In \*\*Seed 2\*\* and \*\*Seed 3\*\*, Layer 6 (Red) and Layer 5 (Green) often converge at high similarity values (approx. 0.85–0.90) by the third task. 3. \*\*Variability Across Seeds:\*\* - While the overall hierarchy (Layer 3 being the lowest) is maintained, the specific ordering of Layers 4, 5, and 6 varies slightly between seeds. For example, in Seed 1, Layer 5 (Green) peaks at 2 Tasks before dropping, whereas in Seed 2 and 3, it remains relatively stable or continues to rise slightly. - The spread of individual data points (the faint dots) is widest at "1 Task" and tends to tighten as more tasks are added, suggesting that the model representations become more consistent and converged as the task load increases. ### Summary The figure demonstrates that as a model is trained on more tasks, its internal representations (especially in deeper layers 4, 5, and 6) become more similar to one another, as indicated by rising CKA scores. Layer 3 remains distinct with significantly lower similarity scores, suggesting it may be capturing more task-specific or lower-level features that do not generalize as broadly as the deeper layers.](figures/Figure13-1.png)
*Figure 13: CKA vs. task count for individual seeds. Each panel shows a different seed. These values are pooled in Fig. 3(d); error bars there represent SEM across seeds.*

in the main text, but broken down by individual seeds. Each panel shows one seed. These per-seed
values are pooled to produce the main text figure, where error bars represent SEM across seeds.
The pattern is consistent across all three seeds: CKA increases substantially from 1 to 2 tasks and
saturates at 2–3 tasks for layers 4–6.


![Figure14-1: Figure 14-1: This figure consists of two line graphs, labeled (a) and (b), which illustrate the Centered Kernel Alignment (CKA) similarity scores across different layers of a neural network as the number of tasks increases. \*\*General Layout and Legend:\*\* Both panels share a common x-axis representing the number of tasks: "1 Task," "2 Tasks," and "3 Tasks." The y-axis for both represents "CKA" similarity, ranging from 0.0 to 1.0. A legend on the far right identifies four colored lines corresponding to different layers of the model: \* \*\*Blue:\*\* Layer 3 \* \*\*Orange:\*\* Layer 4 \* \*\*Green:\*\* Layer 5 \* \*\*Red:\*\* Layer 6 \*\*Panel (a): Across-Task Similarity\*\* This panel shows how similar representations are across different tasks for each layer. \* \*\*Data Points:\*\* Each category on the x-axis features a "swarm" of semi-transparent dots representing individual data points, color-coded by layer. \* \*\*Annotations:\*\* Above the "2 Tasks" column is the text "C(7,2)=21," and above the "3 Tasks" column is "C(7,2)=35" (likely referring to the number of combinations/comparisons made). \* \*\*Trends:\*\* \* \*\*Layer 3 (Blue):\*\* Shows the lowest similarity, starting at approximately 0.2 for 1 task and rising slightly to about 0.4 for 3 tasks. \* \*\*Layer 4 (Orange):\*\* Starts at approximately 0.55 and rises steadily to about 0.78. \* \*\*Layers 5 (Green) and 6 (Red):\*\* These layers show the highest similarity and follow nearly identical paths. They start at approximately 0.65–0.70 for 1 task and rise to a plateau of approximately 0.82–0.85 for 2 and 3 tasks. \* \*\*Key Insight:\*\* Similarity increases as the number of tasks increases, and higher layers (5 and 6) exhibit significantly higher representational similarity across tasks compared to lower layers (3 and 4). \*\*Panel (b): Within-Task Similarity\*\* This panel is titled "Within Task" and specifically focuses on Layer 5 (Green). \* \*\*Data Points:\*\* Similar to panel (a), there are clusters of light green dots representing individual measurements. \* \*\*Trend:\*\* \* For \*\*1 Task\*\*, the CKA similarity is approximately 0.7, with a relatively wide vertical spread of data points (error bar indicates higher variance). \* For \*\*2 Tasks\*\*, the similarity jumps significantly to approximately 0.9, and the data points become much more tightly clustered. \* For \*\*3 Tasks\*\*, the similarity remains stable at approximately 0.9 with low variance. \* \*\*Key Insight:\*\* The internal consistency of representations within a task increases and stabilizes as the model is trained on more tasks, reaching a high level of similarity (0.9) by the time two tasks are involved. \*\*Summary:\*\* The figure demonstrates that as more tasks are added, the neural network's representations become more similar both across different tasks (panel a) and within the same task (panel b). This effect is more pronounced in the deeper layers of the network.](figures/Figure14-1.png)
*Figure 14: Aggregated CKA analysis. (a) CKA vs. task count for single seed, comparing only non-overlapping model pairs (105 pairs for 2-task, 70 pairs for 3-task). (b) Within-task CKA (same task combination, different seeds) increases with task count, indicating multi-task training reduces seed variability.*

Preprint


Figure 12: CKA matrix for two-task models at layer 5. Mean ± SEM across 3 seeds. All pairs
show high alignment (>0.84), substantially higher than single-task models.


Figure 13: CKA vs. task count for individual seeds. Each panel shows a different seed. These
values are pooled in Fig. 3(d); error bars there represent SEM across seeds.
Aggregated CKA Trends.
Fig. 14(a) shows CKA vs. task count for a single seed, using all
 7

=
21 two-task models and all
 7

= 35 three-task models, but only comparing non-overlapping pairs
(models sharing no common tasks). This yields 105 non-overlapping pairs for 2-task models and 70
for 3-task models. Fig. 14(b) shows within-task CKA (same task combination, different seeds) as a
function of task count, demonstrating that multi-task training also reduces seed-to-seed variability:
representations become more consistent not just across tasks but also across random initializations.


Figure 14: Aggregated CKA analysis. (a) CKA vs. task count for single seed, comparing only
non-overlapping model pairs (105 pairs for 2-task, 70 pairs for 3-task). (b) Within-task CKA (same
task combination, different seeds) increases with task count, indicating multi-task training reduces
seed variability.
CKA vs. Generalization (Annotated).
Fig. 15 is an annotated version of Fig. 5(b), with each

![Figure15-1: Figure 15-1: This figure is a scatter plot illustrating the relationship between the representational similarity of models trained on different tasks and the resulting cross-task performance improvement. ### Axes and Scale \* \*\*X-axis:\*\* Labeled "CKA between models trained on only X vs only Y." This axis represents the Centered Kernel Alignment (CKA) score, a measure of similarity between neural network representations. The scale ranges from 0.4 to 1.0, with major tick marks every 0.2 units. \* \*\*Y-axis:\*\* Labeled "Improvement on Y by training on X." This axis measures the transfer learning benefit, specifically how much performance on task Y improves when the model is pre-trained on task X. The scale ranges from 0.0 to 1.0, with major tick marks every 0.2 units. ### Data Points and Annotations The plot contains approximately 30 individual data points, each represented by a colored circle. Each point is labeled with a task pair in the format "X $\rightarrow$ Y," where X is the training task and Y is the evaluation task. The tasks appear to be abbreviated (e.g., A, T, P, Co, I, D, Go). \* \*\*Task Abbreviations:\*\* Based on the labels, tasks include A (likely Action), T (likely Tool), P (likely Part), Co (likely Color), I (likely Interaction), D (likely Depth), and Go (likely Goal). \* \*\*Color Coding:\*\* The points are color-coded by their target task (Y). For example: \* \*\*Blue circles\*\* (Target A): e.g., $T \rightarrow A$, $P \rightarrow A$, $I \rightarrow A$. \* \*\*Purple circles\*\* (Target Co): e.g., $A \rightarrow Co$, $T \rightarrow Co$. \* \*\*Green/Teal circles\*\* (Target I): e.g., $A \rightarrow I$, $P \rightarrow I$. \* \*\*Orange/Tan circles\*\* (Target P): e.g., $A \rightarrow P$, $Co \rightarrow P$. \* \*\*Red/Pink circles\*\* (Target D): e.g., $T \rightarrow D$, $Go \rightarrow D$. ### Statistical Analysis \* \*\*Regression Line:\*\* A dark gray dotted line represents the linear regression fit for the data. The line shows a positive slope, indicating that as the CKA similarity between models trained on tasks X and Y increases, the transfer improvement from X to Y also tends to increase. \* \*\*Statistics:\*\* In the top-left quadrant of the plot, the following statistical values are provided: \* \*\*$R^2 = 0.188$\*\*: This indicates that approximately 18.8% of the variance in improvement on Y can be explained by the CKA similarity between the models. \* \*\*$p = 0.017$\*\*: This p-value suggests that the positive correlation between representational similarity and task improvement is statistically significant (typically $p < 0.05$). ### Key Insights 1. \*\*Positive Correlation:\*\* There is a statistically significant positive relationship between how similar two tasks are in representation space (CKA) and how well training on one task aids performance on the other. 2. \*\*High Similarity/High Improvement:\*\* Task pairs like $A \rightarrow Co$ and $T \rightarrow Co$ show both high CKA (around 0.8) and high improvement (near 0.9-1.0). 3. \*\*Low Similarity/Low Improvement:\*\* Task pairs involving "D" (Depth), such as $D \rightarrow A$ and $D \rightarrow T$, tend to cluster in the lower-left, showing lower CKA scores (0.5-0.6) and very low improvement (0.0-0.2). 4. \*\*Variance:\*\* While the trend is positive, there is significant spread in the data ($R^2 = 0.188$), suggesting that CKA similarity is a meaningful but not sole predictor of transfer learning success. For instance, $P \rightarrow A$ has a high CKA (~0.88) but only moderate improvement (~0.45).](figures/Figure15-1.png)
*Figure 15: Annotated version of Fig. 5(b). Each point is labeled with its (train→eval) task pair. D=distance, T=triangle area, A=angle, Co=compass, I=inside, P=perimeter.*

point labeled by its (train→eval) task pair.

Preprint


Figure 15: Annotated version of Fig. 5(b). Each point is labeled with its (train→eval) task pair.
D=distance, T=triangle area, A=angle, Co=compass, I=inside, P=perimeter.


![Figure16-1: Figure 16-1: This figure presents four heatmaps illustrating the cross-task performance of a model across different random seeds (Seed 1, Seed 2, Seed 3, and Seed 4). The overall visualization is titled "Evaluation Task" at the top, with "Fine-Tuning Task" labeled on the vertical y-axis. ### Layout and Structure The figure is organized into a 2x2 grid of heatmaps. Each heatmap represents a different experimental run identified by its seed number. - \*\*Axes:\*\* Both the x-axis (Evaluation Task) and y-axis (Fine-Tuning Task) are labeled with seven abbreviations representing specific tasks: \*\*D\*\* (Detection), \*\*T\*\* (Tracking), \*\*A\*\* (Action), \*\*Co\*\* (Counting), \*\*I\*\* (Identification), \*\*P\*\* (Pose), and \*\*Cr\*\* (Crowd). - \*\*Color Scale:\*\* A vertical color bar on the right side indicates "Normalized Improvement." The scale ranges from \*\*0.0 (dark red)\*\*, representing low improvement or performance, through \*\*yellow (mid-range)\*\*, to \*\*1.0 (dark green)\*\*, representing high improvement or maximum performance. - \*\*Cell Values:\*\* Each cell in the 7x7 matrices contains a numerical value representing the normalized improvement score. Cells on the main diagonal (where the Fine-Tuning Task matches the Evaluation Task) are marked with a small "T" in the upper-left corner, indicating the target task performance. ### Data Trends and Insights 1. \*\*Diagonal Performance (Self-Transfer):\*\* The diagonal values generally show high performance, often ranging from 0.60 to 1.05. This indicates that models fine-tuned on a specific task perform well when evaluated on that same task. 2. \*\*Cross-Task Transferability:\*\* \* \*\*High Transferability:\*\* Tasks like \*\*A\*\* (Action) and \*\*P\*\* (Pose) frequently show high scores (green cells) when evaluated on other tasks like \*\*Co\*\* (Counting), \*\*I\*\* (Identification), and \*\*Cr\*\* (Crowd). For example, in Seed 3, Fine-Tuning Task \*\*A\*\* achieves a score of 0.98 on Evaluation Task \*\*Co\*\*. \* \*\*Low Transferability:\*\* Task \*\*D\*\* (Detection) consistently shows poor transferability to other tasks, with many red cells in its row (e.g., scores as low as 0.00 or 0.01 for tasks T and A across all seeds). \* \*\*Asymmetric Transfer:\*\* Transfer is often not reciprocal. For instance, while Fine-Tuning on \*\*P\*\* (Pose) transfers well to \*\*D\*\* (Detection) (scores between 0.50 and 0.82), Fine-Tuning on \*\*D\*\* transfers very poorly to \*\*P\*\* (scores between 0.02 and 0.17). 3. \*\*Consistency Across Seeds:\*\* The general patterns of "hot" (green) and "cold" (red) regions are remarkably consistent across all four seeds. This suggests that the observed transferability (or lack thereof) between these specific tasks is a robust property of the model and data, rather than an artifact of random initialization. 4. \*\*Task Clusters:\*\* There appears to be a cluster of high transferability between \*\*Co, I,\*\* and \*\*Cr\*\*, as evidenced by the dense green blocks in the bottom-right quadrants of the matrices. ### Summary of Key Findings The figure demonstrates that while some tasks (like Action and Pose) provide features that are highly beneficial for a wide range of other tasks, others (like Detection) are much more specialized. The high degree of similarity between the four seeds confirms the reliability of these task-transfer dynamics.](figures/Figure16-1.png)
*Figure 16: Single-task fine-tuning results for individual seeds. Per-seed version of Fig. 5(a), organized in a 2×2 grid.*


![Figure17-1: Figure 17-1: This figure presents a series of five heatmaps titled "Evaluation Tasks," illustrating the normalized improvement of various fine-tuning task combinations across different evaluation metrics. The figure is organized into five columns: an "Average" heatmap followed by four individual "Seed" heatmaps (Seed 1, Seed 2, Seed 3, and Seed 4). ### Layout and Axes \* \*\*Vertical Axis (Fine-Tuning Tasks):\*\* The y-axis lists 21 different combinations of fine-tuning tasks. These tasks are represented by abbreviations: D (Deduction), T (Theory of Mind), A (Abductive Reasoning), Co (Contextual), I (Inductive Reasoning), P (Parallelism), and Cr (Creative). The combinations are pairs, such as "D,T," "A,Co," "I,P," etc. Notably, some task labels on the y-axis are highlighted in red (D,T; D,Cr; D,A; D,P; D,Co; D,I), suggesting these specific combinations involve "Deduction" and are of particular interest. \* \*\*Horizontal Axis (Evaluation Tasks):\*\* Each heatmap has seven columns corresponding to the evaluation tasks: D, T, A, Co, I, P, and Cr. \* \*\*Color Scale:\*\* A vertical color bar on the far right indicates "Normalized Improvement." The scale ranges from 0.0 (dark red) to 1.0 (dark green), with yellow representing the midpoint. ### Data Trends and Insights \* \*\*Performance Patterns:\*\* Dark green cells (values near or exceeding 1.0) indicate high improvement, while dark red cells (values near 0.0) indicate low or no improvement. \* \*\*Diagonal/Self-Improvement:\*\* There is a strong trend where fine-tuning on a specific task pair leads to high performance when evaluated on those same tasks. For example, the "A,Co" fine-tuning row shows very high values (dark green) in the "A" and "Co" evaluation columns across all seeds. \* \*\*Cross-Task Generalization:\*\* \* The "Average" heatmap shows that certain combinations, like "A,Co," "T,Co," and "A,P," exhibit broad generalization, with many green cells across various evaluation tasks. \* Conversely, combinations involving "D" (Deduction) often show poor generalization to other tasks, as evidenced by the frequent red and orange cells in rows like "D,Cr," "D,P," and "D,I," particularly in the "T," "A," and "P" evaluation columns. \* \*\*Seed Variability:\*\* While the general patterns are consistent across Seed 1 through Seed 4, there is noticeable stochasticity. For instance, in the "D,T" row, Seed 3 shows significantly higher improvement in the "D" and "T" columns (0.75, 0.68) compared to Seed 2 (0.47, 0.36). \* \*\*Specific Observations:\*\* \* The "A,Cr" and "T,P" combinations appear to be among the most robust, maintaining high green values (0.70 to 1.09) across almost all evaluation tasks and seeds. \* The "D,I" combination consistently performs poorly across most evaluation tasks except for "I" and "Cr," where it shows moderate to high improvement. ### Annotations Each cell contains a numerical value representing the normalized improvement score. Small "T" superscripts are present in many cells, likely indicating a specific statistical threshold or a "Target" task within that fine-tuning pair. In summary, the figure demonstrates that while some fine-tuning task combinations (like those involving Abductive or Theory of Mind reasoning) generalize well across different evaluation metrics, others (particularly those paired with Deduction) show much more localized or inconsistent improvements.](figures/Figure17-1.png)
*Figure 17: Two-task fine-tuning normalized improvement for all 21 task combinations. Leftmost panel shows average across seeds; remaining panels show individual seeds.*

Preprint
E.4


### ADDITIONAL FINE-TUNING EVALUATION RESULTS


Raw fine-tuning results for individual seeds.


Figure 16: Single-task fine-tuning results for individual seeds. Per-seed version of Fig. 5(a),
organized in a 2×2 grid.


Figure 17: Two-task fine-tuning normalized improvement for all 21 task combinations. Leftmost panel shows average across seeds; remaining panels show individual seeds.


![Figure18-1: Figure 18-1: This figure presents a series of heatmaps titled "Evaluation Task," illustrating the performance impacts of multi-task fine-tuning across different random seeds. The figure is organized into four main columns, each representing a different experimental run labeled "Seed 1" through "Seed 4." ### Layout and Axes - \*\*Vertical Axis (Fine-Tuning Tasks):\*\* The y-axis lists 21 distinct pairs of tasks used for fine-tuning. These tasks are represented by abbreviations: D (De-identification), T (Textual Entailment), A (Abbreviation Expansion), Co (Coreference Resolution), I (I2B2 NER), P (PHI Extraction), and Cr (Concept Extraction). Task pairs involving "D" (De-identification) are highlighted in red text on the left margin. - \*\*Horizontal Axis (Evaluation Task):\*\* Each seed contains a sub-grid with seven columns, labeled D, T, A, Co, I, P, and Cr. These represent the individual tasks on which the fine-tuned models are evaluated. - \*\*Color Scale:\*\* A diverging color bar on the right side of the figure indicates the degree of "Synergy" (blue) or "Interference" (red). Dark blue represents high positive values (synergy), white represents zero (neutral), and dark red represents high negative values (interference). ### Data and Trends Each cell in the heatmaps contains a numerical value representing the performance change for a specific evaluation task (column) after being fine-tuned on a specific task pair (row). Small "T" markers are placed above certain cells to denote statistical significance or specific target tasks. 1. \*\*Seed Variation:\*\* There is notable variability across the four seeds. For example, the task pair "D, T" shows mostly neutral to slight synergy in Seed 1 and Seed 3, but significant interference (dark red cells, e.g., -0.53, -0.60) in Seed 2. 2. \*\*Task-Specific Interference:\*\* Task pairs involving "D" (De-identification), marked in red on the y-axis, frequently show rows with significant red shading, particularly in Seeds 2, 3, and 4. This suggests that fine-tuning with the De-identification task often leads to negative transfer (interference) across multiple evaluation tasks. 3. \*\*Synergy Patterns:\*\* Certain task pairs consistently show blue shading (synergy). For instance, the "T, P" and "A, Cr" pairs often result in positive values across several evaluation tasks, particularly in Seed 4, where "T, P" shows values like 0.44 and 0.38. 4. \*\*Diagonal/Target Task Performance:\*\* In many cases, the evaluation tasks that match the fine-tuning tasks (the "target" tasks marked with 'T') show synergy. For example, in Seed 4, the "T, I" fine-tuning pair shows strong synergy for evaluation tasks T (0.32) and I (0.10). ### Key Insights - \*\*Instability of Multi-task Learning:\*\* The significant differences between Seed 1 and Seed 2 highlight that the outcome of multi-task fine-tuning can be highly sensitive to initial conditions or random seeding. - \*\*De-identification as a Disruptor:\*\* The "D" task appears to be a common source of interference, frequently degrading performance on other tasks regardless of the seed. - \*\*Task Compatibility:\*\* Some tasks (like T, P, and A) appear more "compatible" with others, showing more frequent blue cells across the different seeds compared to the "D" task pairs.](figures/Figure18-1.png)
*Figure 18: Deviation from best-teacher expectation for all 21 two-task combinations. All 4 seeds shown; average is in main text Fig. 6(c).*

Preprint


Figure 18: Deviation from best-teacher expectation for all 21 two-task combinations. All 4
seeds shown; average is in main text Fig. 6(c).


![Figure19-1: Figure 19-1: This figure consists of two three-dimensional scatter plots, labeled (a) and (b), which visualize genetic data projected into different coordinate spaces. Both plots use a color-coding scheme that corresponds to the geographical origin of the samples, resulting in a distribution of points that remarkably resembles a world map. \*\*Panel (a): Principal Component Analysis (PCA)\*\* This panel displays the first three principal components (PC1, PC2, and PC3) of a genetic dataset. \* \*\*Axes:\*\* The horizontal axis represents PC1 (ranging from approximately -350 to 350), the vertical axis represents PC2 (ranging from -300 to 100), and the depth axis represents PC3 (ranging from -50 to 300). \* \*\*Data Distribution:\*\* The data points are clustered into distinct groups that mirror global geography. \* On the left (negative PC1), yellow and orange clusters represent the Americas. \* In the center-left, a large blue cluster represents Sub-Saharan Africa. \* Above Africa, red and dark orange clusters represent Europe and North Africa. \* In the center-right, light blue and pink clusters represent the Middle East and Central/South Asia. \* On the far right (positive PC1), purple and green clusters represent East Asia and Oceania. \* \*\*Key Insight:\*\* The PCA effectively captures the primary axes of genetic variation, which are highly correlated with geographic distance and ancestry. \*\*Panel (b): Residual PC and Coordinate Prediction\*\* This panel shows a transformation of the data, likely aiming to align genetic variation more directly with physical geographic coordinates. \* \*\*Axes:\*\* The vertical axis is labeled "y-coordinate prediction" (ranging from -10 to 30). The top horizontal axis is "X-coordinate prediction" (ranging from -60 to 20). The depth axis is labeled "Residual PC1" (ranging from -300 to 300). \* \*\*Data Distribution:\*\* The clusters from panel (a) are rearranged to more accurately reflect a Mercator-like projection of the world. \* The Americas (yellow/orange) are on the left. \* Africa (blue) is in the lower center. \* Europe (red) is in the upper center. \* Asia (pink/purple) and Oceania (green) are on the right. \* \*\*Key Insight:\*\* This plot demonstrates that after accounting for the primary genetic variation (the "Residual PC"), the remaining signal can be used to predict the geographic coordinates (latitude and longitude) of an individual's origin with high accuracy. \*\*Summary Comparison:\*\* While panel (a) shows the raw genetic structure where the "world map" shape is an emergent property of the top three principal components, panel (b) explicitly maps these genetic components onto a spatial coordinate system. Both panels highlight the strong correlation between human genetic diversity and geographic location.](figures/Figure19-1.png)
*Figure 19: Representations when Atlantis is included during pretraining. (a) PCA projection showing Atlantis cities (small cluster in Atlantic region) integrated with world cities. (b) Linear probe reconstruction confirming geographic accuracy. Unlike fine-tuned models, Atlantis cities lie on the same manifold as other cities.*

Preprint
E.5


### PRETRAINING VARIATIONS


Pretraining with Atlantis.
In the main text, we showed that fine-tuning on divergent tasks fails
to integrate Atlantis cities into the learned representation manifold (Fig. 6d, red histogram). To
verify that this failure stems from fine-tuning dynamics rather than a peculiarity of the geometry
around Atlantis, we trained a model with Atlantis cities included from the start of pretraining. Fig. 19 shows the resulting representations: Atlantis cities are seamlessly integrated into the
world manifold, indistinguishable from other cities in both PCA projections (a) and linear probe reconstructions (b). This confirms that the representation space can readily accommodate Atlantis,
and thus, the integration failure observed in fine-tuning is a property of the optimization dynamics,
not a fundamental limitation of the architecture or task.


Figure 19: Representations when Atlantis is included during pretraining. (a) PCA projection
showing Atlantis cities (small cluster in Atlantic region) integrated with world cities. (b) Linear
probe reconstruction confirming geographic accuracy. Unlike fine-tuned models, Atlantis cities
lie on the same manifold as other cities.
Wider Model.
To test whether our findings depend on model capacity, we trained a wider model
with 2× the hidden dimension (256 vs. 128) and intermediate size (1024 vs. 512), resulting in
approximately 4× the parameters. Fig. 20 shows fine-tuning results for this wider model: (a) singletask fine-tuning normalized improvement; (b) two-task fine-tuning normalized improvement; (c)

![Figure20-1: This figure consists of three heatmaps, labeled \*\*a\*\*, \*\*b\*\*, and \*\*c\*\*, illustrating the "Normalized Improvement" of various metrics or conditions across different categories. The categories are represented by single-letter or double-letter abbreviations: D, T, A, Co, I, P, and Cr. ### Color Scale and Legend A vertical color bar is positioned to the right of panel \*\*a\*\*, serving as the legend for panels \*\*a\*\* and \*\*b\*\*. The scale ranges from \*\*0.0 (dark red)\*\* to \*\*1.0 (dark green)\*\*, with a neutral \*\*light yellow\*\* at the midpoint. The label reads "Normalized Improvement." Panel \*\*c\*\* uses a different color scheme, ranging from \*\*dark red (negative values)\*\* to \*\*dark blue (positive values)\*\*, centered at white (zero). ### Panel (a): Single Category Matrix This is a 7x7 square matrix where both the rows and columns are labeled with the single letters: \*\*D, T, A, Co, I, P, Cr\*\*. \* \*\*Diagonal Values:\*\* The diagonal cells (where row and column labels match) are marked with a small "T" in the upper-left corner. These values are: D-D (0.58), T-T (0.69), A-A (0.68), Co-Co (0.99), I-I (0.89), P-P (0.81), and Cr-Cr (0.98). \* \*\*General Trends:\*\* The "Co" (Column 4) and "Cr" (Column 7) columns show high improvement scores (mostly green, values > 0.65) across most rows. Conversely, the "D", "T", and "A" columns (Columns 1-3) show lower improvement scores (mostly red/orange, values < 0.50) for the top three rows. \* \*\*Highest Values:\*\* The highest improvement is seen at the intersection of Co-Co (0.99) and Cr-Cr (0.98). ### Panel (b): Combined Category Matrix This matrix has 6 rows and 7 columns. The columns remain \*\*D, T, A, Co, I, P, Cr\*\*. The rows represent combinations: \*\*D,Cr; D,P; D,Co; A,Co; T,Co; I,Cr\*\*. \* \*\*Row Performance:\*\* The row \*\*A,Co\*\* shows consistently high performance across all columns, with values ranging from 0.59 to 1.09 (dark green). \* \*\*Column Performance:\*\* Columns \*\*Co\*\* and \*\*Cr\*\* continue to show high values (mostly dark green) across almost all combined rows. \* \*\*Specific High Points:\*\* The value at (A,Co) row and (A) column is 1.09, and (I,Cr) row and (Cr) column is 1.03, both exceeding the nominal 1.0 scale. ### Panel (c): Comparative Matrix This matrix follows the same 6x7 structure as panel \*\*b\*\*. The color scale here is blue-to-red. \* \*\*Negative Trends (Red):\*\* The first three rows (\*\*D,Cr; D,P; D,Co\*\*) are dominated by negative values (shades of red), particularly in columns T through Cr, with values ranging from -0.01 to -0.42. \* \*\*Positive Trends (Blue):\*\* The row \*\*A,Co\*\* shows the strongest positive values in the first three columns: D (0.43), T (0.28), and A (0.41), indicated by dark blue. \* \*\*Neutral Trends (Light Blue/White):\*\* The bottom two rows (\*\*T,Co\*\* and \*\*I,Cr\*\*) show values close to zero or slightly positive (0.01 to 0.26), represented by very light blue or white. ### Key Insights 1. \*\*Consistency of "Co" and "Cr":\*\* Across all panels, the categories "Co" (likely Coordination or Correlation) and "Cr" (likely Criterion or Credit) consistently yield the highest normalized improvement scores. 2. \*\*Synergy in Combinations:\*\* Panel \*\*b\*\* suggests that combining categories (like A,Co) can lead to improvements exceeding the baseline of single categories. 3. \*\*Differential Impact:\*\* Panel \*\*c\*\* highlights that while some combinations (A,Co) provide a positive comparative advantage in certain metrics (D, T, A), others (D,Cr; D,P) result in a relative decrease in performance across most metrics.](figures/Figure20-1.png)
*Figure 20: Fine-tuning results for wider model (2× hidden dimension). For all panels: rows = fine-tuning task(s), columns = evaluation task. (a) Single-task fine-tuning normalized improvement. (b) Two-task fine-tuning normalized improvement. (c) Deviation from best-teacher expectation; distance-containing combinations (red labels) still show degraded generalization.*

deviation from best-teacher expectation. We still observe that distance-containing combinations
(red labels in panel c) show degraded cross-task generalization. This suggests that divergent task
interference is not simply a capacity limitation.
F


### EXTENDED RELATED WORK


See Sec. 2 for main related work.
Internal Representations.
Understanding internal representations has roots in neuroscience
(Hubel & Wiesel, 1962), informing early neural network development (Fukushima, 1980; Bengio
et al., 2014; Rosenblatt, 1958; Rumelhart et al., 1986). Recent work has revealed that language models develop structured “world models” encoding geographic, temporal and relational information (Li
et al., 2022; Gurnee & Tegmark, 2023; Nanda et al., 2023b; Marks & Tegmark, 2024), with similar
representations emerging during in-context learning (Vafa et al., 2025). Mechanistic interpretability
and sparse autoencoders have enabled decomposition of neural activations into interpretable features (Anthropic AI, 2023; Templeton et al., 2024). Researchers have also uncovered that models

Preprint


Figure 20: Fine-tuning results for wider model (2× hidden dimension). For all panels: rows =
fine-tuning task(s), columns = evaluation task. (a) Single-task fine-tuning normalized improvement.
(b) Two-task fine-tuning normalized improvement. (c) Deviation from best-teacher expectation;
distance-containing combinations (red labels) still show degraded generalization.
represent meaningful properties of data—concepts (Pearce et al., 2025; Higgins et al., 2017), features (Olah et al., 2017), and abstractions (Lee et al., 2025; Arditi et al., 2024)—in interpretable
ways. Furthermore, PRH posits that diverse models converge toward similar representational structures (Huh et al., 2024). However, recent work questions this representational optimism, suggesting
that deep network representations may be more brittle than previously assumed (Kumar et al., 2025).
Only recent work has begun examining how representations emerge during pretraining in real LLMs
(Li et al., 2025a; Ge et al., 2025) or how they change during fine-tuning (Lee et al., 2024). Our work
takes a complementary perspective, studying the factors that control the formation of these representations and how networks integrate new entities into their representation space via fine-tuning.
Fine-tuning.
The pretraining-finetuning paradigm has become central to modern deep learning,
with seminal works establishing its effectiveness in computer vision (Krizhevsky et al., 2012; He
et al., 2015) and natural language processing (Devlin et al., 2018; Radford et al., 2018). Despite
widespread success, fine-tuning exhibits poorly understood behaviors such as the reversal curse
(Berglund et al., 2024; Lampinen et al., 2025), out-of-context reasoning limitations (Treutlein et al.,
2024), and off-target effects (Betley et al., 2025). On this background, careful studies of fine-tuning
and other low-compute adaptation methods have raised pessimism about whether models can learn
fundamentally new abilities, suggesting they may merely form “thin wrappers” around pretrained
representations (Jain et al., 2023; Ward et al., 2025; Yue et al., 2025; Qin et al., 2025; Zhao et al.,
2025; Zweiger et al., 2025). Fine-tuning has also been studied across diverse directions: parameter
efficiency (Hu et al., 2021; Lester et al., 2021), zeroth-order optimization (Malladi et al., 2024),
weight composition (Ilharco et al., 2023), and representation adaptation (Wu et al., 2024). Work
on feature distortion (Kumar et al., 2022) is perhaps most related to ours, though representational
changes are assumed rather than directly measured. Our work examines this question in a controlled
setup where ground-truth world structure enables precise measurement of representation adaptation.
Dynamics of Representations.
Recent work has begun studying how representations evolve during in-context learning (Shai et al., 2025; Demircan et al., 2024) or fine-tuning (Casademunt et al.,
2025; Minder et al., 2025). Relatedly, Lubana et al. (2025) show that representations exhibit rich
temporal dynamics that standard interpretability methods (e.g., SAEs) fail to capture due to stationarity assumptions. Fu et al. (2025) show that VLMs trained by merging LLMs and vision encoders
often fail to utilize representations surfaced by the vision encoder, i.e. the representations exist but
remain unused.
Geometric Deep Learning.
Geometric deep learning studies how data geometry interacts with
model architectures, developing equivariant networks that respect symmetries (Bronstein et al.,
2021; Cohen & Welling, 2016; Weiler & Cesa, 2021). While our world is defined on a 2D plane,
one might ask: why not a sphere, torus, or other manifold? This is an interesting direction, but not
our focus. We study how neural networks adapt internal representations to tasks in an arbitrarily
chosen geometry. Moreover, a change in world geometry can be absorbed into the task definition
(e.g., geodesic vs. Euclidean distance), so the key question remains how representations form given

Preprint
the task, not the underlying manifold. Planar coordinates also allow clean linear probing of world
representations. Our models are standard transformers without geometric priors; we study what
representations emerge purely from training on task data, treating geometry as emergent rather than
imposed.
Loss Plateaus.
Our crossing task fails to learn in single-task training despite escaping an initial plateau (likely output format learning), suggesting it remains stuck in a deeper plateau. Such
plateaus are notoriously difficult for transformers. Recent work has studied this phenomenon mechanistically in transformers (Hoffmann et al., 2024; Gopalani & Hu, 2025; Singh et al., 2024), while
others relate it to more general optimization challenges in deep learning such as simplicity bias and
gradient starvation (Shah et al., 2020; Pezeshki et al., 2021; Bachmann & Nagarajan, 2025). Most
related to our findings, Kim et al. (2025) show that multi-task training shortens loss plateaus, similar
to why our crossing task trains successfully when joined with any other task.

