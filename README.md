# LSTM for reconstructing spatial trajectories

LSTM model for reconstructing spatial trajectories in polar coordinates (circular, spiral, Lissajous). Scripts are available for data creation and visualization, LSTM model training, export of TorchScript wrappers compatible with the [conTorchionist](https://github.com/ecrisufmg/contorchionist) library, autoregressive testing, and visual validation of the generated trajectories.

```mermaid
graph TD
classDef inputs fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
classDef layers fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
classDef operations fill:#e0f2f1,stroke:#00695c,stroke-width:2px;
classDef output fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

subgraph ProcessamentoTemporal["Processamento Temporal"]
InputSeq["Input Sequence<br/>(B,10,8)"]:::inputs
LSTMLayer["LSTM<br/>(2 layers, hidden=80)"]:::layers
LastState["Last Hidden State<br/>(B,80)"]:::operations
InputSeq --> LSTMLayer --> LastState
end

subgraph CondicionamentoTipo["Condicionamento por Tipo"]
InputType["Type ID<br/>(B,1)"]:::inputs
EmbedLayer["Embedding<br/>(3 → 16)"]:::layers
TypeVec["Type Vector<br/>(B,16)"]:::operations
InputType --> EmbedLayer --> TypeVec
end

Concat["Concat<br/>(80 + 16 = 96)"]:::operations
FC1["Linear FC1<br/>(96 → 32)"]:::layers
FC2["Linear FC2<br/>(32 → 16)"]:::layers
FC3["Linear FC3<br/>(16 → 4)"]:::layers
Tanh["Tanh<br/>(-1 a 1)"]:::layers
FinalOut["Predicted<br/>(r, sin, cos, dur)"]:::output

LastState --> Concat
TypeVec --> Concat
Concat --> FC1 --> FC2 --> FC3 --> Tanh --> FinalOut
```
