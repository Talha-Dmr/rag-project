# Benchmark Results and Evaluation Metrics

## Final Targeted Benchmark

The final benchmark contains 160 financial regulation questions. It was designed as a targeted hallucination stress test after analyzing where the baseline RAG system failed. The benchmark focuses on cases where a model may accept false premises, transfer evidence across regulators, or invent precise operational requirements from partial evidence.

<table>
  <thead>
    <tr>
      <th>System Variant</th>
      <th>Expected Behaviour Match</th>
      <th>Answer Rate</th>
      <th>Abstain Rate</th>
      <th>Forbidden Claim Hit Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Baseline RAG</td>
      <td>83.125%</td>
      <td>100.000%</td>
      <td>0.000%</td>
      <td>0.625%</td>
    </tr>
    <tr>
      <td>RAG + Detector</td>
      <td>95.625%</td>
      <td>49.375%</td>
      <td>50.625%</td>
      <td>0.625%</td>
    </tr>
    <tr>
      <td>RAG + Detector + Stochastic Gate</td>
      <td>98.125%</td>
      <td>45.000%</td>
      <td>55.000%</td>
      <td>0.000%</td>
    </tr>
  </tbody>
</table>

## Benchmark Category Distribution

<table>
  <thead>
    <tr>
      <th>Question Category</th>
      <th>Count</th>
      <th>Evaluation Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>cross_source_nuanced</td>
      <td>72</td>
      <td>Tests whether the system rejects unsupported transfer of evidence between different regulators, documents, or supervisory contexts.</td>
    </tr>
    <tr>
      <td>low_evidence_policy</td>
      <td>40</td>
      <td>Tests whether the system avoids inventing exact operational details when the evidence only supports a broad topic.</td>
    </tr>
    <tr>
      <td>false_premise</td>
      <td>32</td>
      <td>Tests whether the system rejects misleading questions that assume a fabricated regulatory requirement is true.</td>
    </tr>
    <tr>
      <td>factual_supported</td>
      <td>16</td>
      <td>Checks that the system can still answer normally when the requested information is directly supported by retrieved evidence.</td>
    </tr>
  </tbody>
</table>

## Metric Definitions

<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Meaning</th>
      <th>Why It Matters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Expected Behaviour Match</td>
      <td>The percentage of questions where the system performed the intended action: answer when evidence was sufficient, qualify or refuse when evidence was weak, and reject false premises.</td>
      <td>This is the main end-to-end quality metric. It evaluates whether the full system behaves correctly, not only whether it produces fluent text.</td>
    </tr>
    <tr>
      <td>Answer Rate</td>
      <td>The percentage of questions where the system returned a substantive answer instead of abstaining.</td>
      <td>A high answer rate shows usefulness, but it can be risky if the system answers unsupported questions too confidently.</td>
    </tr>
    <tr>
      <td>Abstain Rate</td>
      <td>The percentage of questions where the system refused to answer because the available evidence was insufficient or risky.</td>
      <td>In financial regulation, abstaining can be safer than providing an unsupported answer. This metric measures how often the safety layer blocks uncertain responses.</td>
    </tr>
    <tr>
      <td>Forbidden Claim Hit Rate</td>
      <td>The percentage of questions where the generated answer included a claim that was explicitly marked as unsupported or fabricated in the benchmark.</td>
      <td>This directly measures harmful hallucination behaviour. Lower is better, and 0.000% means the system avoided all benchmark-defined forbidden claims.</td>
    </tr>
  </tbody>
</table>

## Category Explanations

### cross_source_nuanced

These questions contain evidence from more than one document or regulator. The risk is that the model may incorrectly transfer a rule from one authority to another. For example, related EBA evidence should not automatically become a PRA or ECB requirement.

### low_evidence_policy

These questions provide evidence that supports a general regulatory topic but not a precise operational detail. The correct behaviour is to qualify the answer or abstain, rather than inventing details such as fixed deadlines, portal names, templates, or mandatory software products.

### false_premise

These questions include a misleading assumption, such as claiming that a document requires a fabricated rule. The system should reject the premise or abstain instead of accepting the false requirement.

### factual_supported

These are sanity-check questions where the answer is directly supported by the retrieved evidence. They prevent the system from appearing safe only because it refuses too often.

## Key Finding

The baseline RAG system answered every question, which made it useful but more vulnerable to overclaiming. Adding the detector substantially improved expected behaviour by filtering unsupported answers. Adding stochastic uncertainty gating produced the strongest safety result by further reducing risky outputs and eliminating forbidden claim hits in the final benchmark.

