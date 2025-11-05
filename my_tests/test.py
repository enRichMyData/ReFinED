from my_tests.htr_json_test import eval_el_challenge
from my_tests.utility.test_utils import load_model
from my_tests.accuracy import evaluate_refined
from refined.data_types.base_types import Entity, Span

device = "gpu"
entity_set = "wikidata"
model = "wikipedia_model_with_numbers"

refined_model = load_model(device=device, entity_set=entity_set, model=model)

# ------- Run official evaluation -------
metrics = evaluate_refined(refined_model, "HTR1")

# ------- Run HTR json test -------
all_spans, truths, duration = eval_el_challenge(
    model=refined_model,
    eval_set="HTR1",
    entity_set=entity_set,
    batch_size=512,
    prediction_mode="cell",
    verbose=False
)

def generate_samples(n=10, correct_prob=0.6, no_pred_prob=0.2):
    import random
    pred_spans = []
    truths = []

    for i in range(n):
        truth_qid = f"Q{i+1}"
        truths.append((0, i, [truth_qid]))

        r = random.random()
        if r < no_pred_prob:
            # no prediction
            pred_spans.append([])
        elif r < no_pred_prob + correct_prob:
            # correct prediction
            pred_spans.append([Span(text=f"Cell{i+1}", start=i*6, ln=5, predicted_entity=Entity(wikidata_entity_id=truth_qid))])
        else:
            # wrong prediction
            wrong_qid = f"Q{n + i + 1}"  # ensure it's different from truth
            pred_spans.append([Span(text=f"Cell{i+1}", start=i*6, ln=5, predicted_entity=Entity(wikidata_entity_id=wrong_qid))])

    return pred_spans, truths


# pred_spans, truths = generate_samples(50)
# accuracy = measure_accuracy(pred_spans=pred_spans, truths=truths, all_metrics=True, verbose=True)