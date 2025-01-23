# VecDB
VectorDB class inherited by SentenceEvaluator

## Examples

* Initialize
  
```python
from VecDB import VecDict
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vecdict = VecDict(sent_emb_model=model)
```

* Setting Vector DB

```python
vecdict.set_vec_dict(documents) # You can use a list of strings for documents here.
vecdict.set_query_answers(question_answer_pairs) # You can use question-answer document pairs.
```

Although `set_query_answers` gets question-answer document pairs directly, still the method saved indices only. So you guys don't really have to worry about RAM bottlenecks.

* Training sentence embedding with vectorDB

What you should do is just put VecDict object as an evaluator.
```python
# ...
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=vecdict, # here
)
trainer.train()

```

* Directly use trained model

```python
vecdict.reset_model(trainer.model) # Don't forget that trainer to return the best model, using `load_best_model_at_end` in `SentenceTransformerTrainingArguments`.
scores, result = vecdict.return_topk()
```

## Why did you write this?

**To directly customize separated tasks, when training task and evaluation task are different.**

* Since contrastive learning, usually sentence embedding is regarded as training task, is not the objective task. What people (including me) want is vector retrieval performance. Sentence-Transformers, really popular open library also noticed this and separated evaluation tasks. Like, **RerankingEvaluator**, or **InformationRetrivalEvaluator** are already implemented.
* But implementing evaluator and using trained model is not that smoothly connected. For example, you guys still have to detach model from gpu, load trained model to gpu again, infer model to the vector DB system, test the score there and try train in here again... I wanted to unite these phases.
* So this is what I made.
