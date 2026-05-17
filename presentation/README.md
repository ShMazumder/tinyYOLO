# Anti-Gravity / Living Presentation Artifact Notes

This document accompanies the live interactive presentation designed for the **TinyYOLO R1 Revision**. It outlines the core concept, visual layouts, and typographic hierarchies implemented to achieve an Apple/DeepMind-level aesthetic of calm authority and high-negative-space precision.

## The Living Presentation Walkthrough

The browser subagent successfully executed a walkthrough of the slide deck, advancing through all 5 slides to trigger their custom visual transitions. Below is the live animation of the walkthrough:

![Walkthrough of the Living Presentation](./presentation_flow.webp)

> [!NOTE]
> **To Open the Presentation Directly:**
> You can open the interactive file directly in your browser:
> [presentation.html](file:///Applications/XAMPP/xamppfiles/htdocs/tinyYOLO/presentation/presentation.html)
> *Control the slides using the bottom-right arrows, pagination dots, or your keyboard **Left / Right arrow keys**.*

---

## Slide-by-Slide Design Breakdown

````carousel
### Slide 1: Thinking Becoming Structure
*   **Narrative:** Visualizes the transition from peer review ambiguity ("scribble") to technical code and manuscript structure ("inevitability").
*   **Copy:** *"This presentation is not a static deck. It is a living artifact visualizing the process of turning peer review ambiguity into rigorous, publication-grade academic structure."*
*   **Visual Metaphor:** A dynamic SVG outline scribble representing unstructured critique, flowing through a directional arrow into a balanced, clean 3D-shadowed card representing clean architecture.
<!-- slide -->
### Slide 2: Decanting Critique into Action
*   **Narrative:** Introduces how textual review comments are parsed and explicitly mapped to parameterized codebase edits.
*   **Copy:** *"Our agent read, parsed, and mapped reviewers' concerns directly to file locations, establishing explicit pathways for resolving methodological inconsistencies."*
*   **Visual Metaphor:** A mock code-editor interface rendering a configurable activation (`act='silu'`) in `TinyDetect` with a pulsing accent-blue cursor illustrating agent action on heads.py.
<!-- slide -->
### Slide 3: The LitePAN Pathway
*   **Narrative:** Illustrates the parameter-efficient bidirectional neck pathways of LitePAN.
*   **Copy:** *"Depthwise separable convolutions minimize parameters in our multi-scale neck, enabling seamless spatial and semantic feature fusion at a fraction of standard PAN costs."*
*   **Visual Metaphor:** A clean node diagram with directional connecting arrows outlining the feature flow from P3, P4, P5 down through the lateral projections and FPN pathways.
<!-- slide -->
### Slide 4: Resolving the mAP Paradox
*   **Narrative:** Resolves the apparent VOC mAP performance discrepancy through rigorous mathematical formulation.
*   **Copy:** *"Lightweight models suffer severe precision drop-offs in early epochs. Evaluating tiny architectures under legacy coarse metrics creates highly misleading performance gaps."*
*   **Visual Metaphor:** A LaTeX equation card featuring the legacy 11-point coarse interpolation formula, paired with horizontal comparative graphs illustrating TinyYOLO's performance gains (+1.8% over YOLO-Fastest).
<!-- slide -->
### Slide 5: The Living Deliverables
*   **Narrative:** Celebrates the completely resolved, synchronized, publication-ready submission materials.
*   **Copy:** *"This revision does not live as code changes alone. It is represented as a structured, fully-synchronized set of submission materials ready for top-tier Q1 venues."*
*   **Visual Metaphor:** Sleek capability cards detailing the consolidated manuscript (`report.md`) and the point-by-point rebuttal (`rebuttal_letter.md`) with clean, outline-based document and mail icons.
````

---

## Implemented Design System

> [!TIP]
> **1. Negative Space & Canvas**
> We maintained absolute canvas discipline. The primary background is pure `#ffffff` white, which creates high breathing space and maximum contrast. Flowing radial gradients (`blue` $\to$ `cyan` $\to$ `violet`) are set to an ultra-low `0.15` opacity and pushed to the extreme top-left and bottom-right edges to suggest energy fields/anti-gravity without cluttering content.

> [!IMPORTANT]
> **2. Typographic Hierarchy**
> We used **Outfit** (sans-serif with clean, rounded geometry) for headlines to convey modern, calm authority. Body copy utilizes **Inter** (highly legible, precise letter-spacing) at light/medium weights. No bold blocks or heavy margins were used; each slide strictly presents one headline, one short explanatory sentence, and one thin-line bullet card.

> [!WARNING]
> **3. Color Restraint**
> Colors are used strictly for semantic function, not decoration. The core text uses deep off-black (`#111111`) and secondary labels use charcoal gray (`#666666`). The calm accent blue (`#0066cc`) is reserved exclusively for the most critical elements: key focal words in headlines, navigation highlights, math bar comparative fill, and the pulsing browser cursor.
