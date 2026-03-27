# ICML Rebuttal Repository (Paper ID: 17140)

**To the Respected Reviewers,**

Thank you very much for your time and the constructive feedback provided in your reviews. We sincerely appreciate the opportunity to share our implementation details with you during this rebuttal phase. In response to your comments, we have prepared this anonymous repository to provide additional technical clarity on the implementation and architecture of our work.

**Our Commitment to Open Source:** We believe that transparency and reproducibility are essential for the advancement of the machine learning and embodied AI communities. Parts of the testing environments used in this work are built upon our previously published open-source projects/benchmarks. To strictly adhere to the double-blind policy and avoid identity leakage through self-citation, we have provided the **overall code architecture** and **representative task scenarios** in this repository for verification. 

Upon acceptance, we are committed to properly citing our prior work and releasing the **complete, production-ready codebase**, including all task environments, full training scripts, pre-trained model weights, and comprehensive guides. We would be deeply grateful if given the opportunity to share our research with the community through the ICML platform, and we are eager to support our fellow researchers in exploring and building upon this work in the future.

---

## Code Architecture

The core functional structure of the project:

* **`g1/`**: Houses the URDF/USD and mesh files for the humanoid robots, ensuring the simulation setup is transparent and verifiable. 
* **`halpo_algorithm/`**: Contains the primary implementation and logical framework of our Heterogeneous-Agent Lyapunov Policy Optimization.
* **`source/`**: Defines the task environment structures and configuration files for our benchmarks. Also, it includes the necessary framework wrappers and dependency modules to support the repository's architecture.
* **`wbc_integration/`**: Contains code for the integration of low-level whole body controller.

---

## Release Roadmap & TODO List

To respect the double-blind policy while ensuring technical clarity, we have released the core architecture during the ICML rebuttal process. We are committed to the following release schedule:

- [x] **Algorithm Architecture**: Implementation framework of HALyPO project.
- [x] **Representative Task Scenario**: A functional task configuration and environment instance.
- [x] **Robot Assets**: URDF/USD models and mesh files for humanoid robots.
- [ ] **Full Benchmark Suite and Pre-Trained Models** (*Release upon acceptance*): All collaborative task environments and optimized checkpoints for immediate evaluation.
- [ ] **Complete Training Pipelines** (*Release upon acceptance*): End-to-end training scripts and hyperparameter sweep logs.
- [ ] **Documentation and Deployment** (*Coming Soon*): Step-by-step installation guide and sim-to-real deployment tutorials.

---

## Anonymity Notice

In strict accordance with the ICML double-blind policy, this repository has been thoroughly scrubbed. All author names, institutional affiliations, and identifying metadata have been removed or pseudonymized.
