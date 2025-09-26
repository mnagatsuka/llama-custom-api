# Comparison of EOS Suppression Patterns in Llama.cpp

Below is a comparison of three implementation patterns for controlling the probability of EOS (End of Sequence) after Softmax. It is kept simple and non-redundant.

## Pattern A: `ignore_eos`

* **Description**: Completely ignores the EOS token and forces generation to continue until the specified minimum length.
* **Pros**: Very easy to enforce a minimum length. Guarantees no short outputs.
* **Cons**: Tends to produce verbose or unnatural text. Requires a second request after reaching minimum length.
* **Use Case**: When “short outputs are absolutely unacceptable.”

## Pattern B: `logit_bias`

* **Description**: Adds a negative bias to the EOS logit to reduce its probability. Bias is removed once the minimum length is reached.
* **Pros**: Good balance between naturalness and control. Adjustable strength (e.g., −8 to −12). Simple to implement.
* **Cons**: Works as “strong guidance,” not absolute enforcement. Short outputs may still occur.
* **Use Case**: When naturalness is prioritized, but short outputs should be strongly discouraged.

## Pattern C: Custom LogitsProcessor

* **Description**: Dynamically modifies logits during step-by-step generation. For example: strongly suppress EOS until reaching minimum length, then remove suppression.
* **Pros**: Fast responses (single request), smooth SSE streaming, highest flexibility in control.
* **Cons**: Highest implementation and maintenance cost.
* **Use Case**: When strict control and optimal streaming UX are required.

## Recommended Summary

* **For naturalness & simplicity** → Pattern B
* **For strict minimum length guarantee** → Pattern A
* **For fastest response + streaming UX + precise control** → Pattern C
