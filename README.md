# ICML Rebuttal Repository (Paper ID: 17140)

**To the Respected Reviewers,**

Thank you very much for your time and the constructive feedback provided in your reviews. We sincerely appreciate the opportunity to share our implementation details with you during this rebuttal phase.

In response to your comments, we have prepared this anonymous repository to provide additional technical clarity on **HALyPO**. We understand that a research paper is a continuous process of refinement, and your suggestions have been vital in helping us identify areas where the presentation or technical depth could be improved.

---

## 🏗️ Code Architecture

To respect your review time, we have included the core functional subset of the project that directly addresses the mechanisms discussed in the paper:

* **`halpo_algorithm/`**: Contains the primary implementation of our Heterogeneous-Agent Lyapunov Policy Optimization. This includes the **Lyapunov projection logic** and decentralized update routines which are central to our theoretical contribution.
* **`source/isaaclab_tasks/`**: Defines the task environments and configuration files for our benchmarks (OSP, SCT, and SLH). These files detail the physics parameters and reward structures used in our experiments.
* **`assets/`**: Houses the URDF and mesh files for the humanoid robot and collaborative objects, ensuring the simulation setup is transparent.
* **`source/isaaclab*`**: Includes the necessary framework wrappers and dependency modules to ensure the codebase is as self-contained and reproducible as possible.

---

## 🌿 Our Commitment to Open Source

We believe that transparency and reproducibility are essential for the advancement of the robotics and MARL communities. 

* **Continuous Improvement**: We are currently refining the code and expanding the documentation based on the insights gained from your reviews.
* **Full Release Plan**: If this work is accepted, we are committed to releasing the **complete, production-ready codebase**, including all training scripts, pre-trained model weights, and comprehensive "getting started" guides to help other researchers build upon our work.

---

## 🛡️ Anonymity Notice

In strict accordance with the ICML double-blind policy, this repository has been thoroughly scrubbed. All author names, institutional affiliations, and identifying metadata have been removed or pseudonymized.

**We sincerely hope this code provides the clarity you are seeking. We remain fully open to any further suggestions you may have to make this work more robust and impactful.**
