# Development Roadmap

## Cost Control

- Keep API Keys out of git & GitHub.

## Ingestion

1. [ ] Use [*Docling*](https://docling-project.github.io/docling/getting_started/quickstart/) to parse HTML file 

| Hint | Reason |
|---|---|
|use Python, not CLI| Python code is faster to integrate into our pipeline |
|save DoclingDocument as html under `tmp/html/`| to visually check parsing stage | 
|save DoclingDocument as json under `tmp/docling/`| to later reuse without parsing again |

2. [ ] Use *Docling*'s HybridChunker to split the parsed file into chunks

| Hint | Reason |
|---|---|
| You can experiment with HierarchicalChunker or with different parameters to HybridChunker | An ideal chunk split depends on context - a balance between size and specificity |

3. [ ] Each **chunk** should have associated metadata with title, chaper and article

| Hint | Reason |
|---|---|
| e.g. title: 'Dispozitii generale', title_no: 1, chapter: 'Definitii', chapter_no: 3, article: 'Definitia sediului permanent', article_no: 8 | metadata can be used to later build useful citations for the user, and to improve search method |

4. [ ] Create **embeddings** from each chunk using *Hugging Face*'s sentence-transformers/all-MiniLM-L6-v2
5. [ ] Save **embeddings** in *ChromaDB* vector store. (save on disk using SQLite)

## Retrieval-Augment-Generate

1. [ ] Create LangGraph node with name `Retriever` that performs search in vector store
| Hint | Reason |
|---|---|
| Save chunk metadata in AgentState | metadata can be used to later build useful citations for the user|

2. [ ] Create LangGraph node with name `Generator` that receives chunk contents and user question and returns the answer.
3. [ ] Build a graph: START -> Retriever -> Generator -> END.

## Monitor

1. [ ] Install Arize Phoenix and connect to LangGraph code

## Testing 

1. [ ] Test it.

## Evaluation

1. [ ] Create evaluation pipeline and dataset

## Try alternative graphs

1. [ ] Create Gatherer Node that calls Retriever node until it has sufficient context to answer.
2. [ ] Try a different graph: START -> Gatherer ReAct Loop ( -- Retriever) --> Generate --> END.
3. [ ] Compare evaluation scores, cost and latency against first graph

| Hint | Reason |
|---|---|
| use MAX_TURNS for loop | keep costs and latency under control |

## UI

1. [ ] Create a chatbot page that answers user's questions
2. [ ] Add citations side-panel when clicking on `See sources` button next to each answer

