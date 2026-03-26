📁 ICML Rebuttal Repository (Paper ID: 17140)

To the Respected Reviewers,

Thank you for your time and for the detailed feedback provided in your reviews. We sincerely appreciate the opportunity to share our implementation with you during this rebuttal phase.

In response to your comments, we have prepared this anonymous repository to provide more clarity on the technical details of HALyPO. We understand that a research paper is a continuous process of refinement, and your suggestions have been vital in helping us identify areas for improvement.

🏗️ Code Overview

To respect your time, we have included a core subset of the project that directly addresses the mechanisms discussed in the paper:

    halpo_algorithm/: Contains the primary implementation of our Lyapunov-based policy optimization. We hope this clarifies the gradient rectification and projection logic.

    source/isaaclab_tasks/: Includes the environment configurations for the collaborative tasks (OSP, SCT, and SLH). These define the physics and reward structures used in our benchmarks.

    assets/: Provides the robot and object models used in the Isaac Sim environment.

    source/isaaclab*: Necessary framework modules included to ensure the code structure is as self-contained as possible.

🌿 Our Commitment to the Community

We believe that open-source transparency is essential for reproducible robotics research.

    Continuous Improvement: We are currently working on further cleaning the code and expanding the documentation based on your feedback.

    Full Release: If this work is accepted, we are committed to releasing the complete codebase, including all training scripts, pre-trained weights, and a comprehensive "getting started" guide to help other researchers reproduce our results.

🛡️ Anonymity Note

This repository has been carefully scrubbed to comply with ICML's double-blind policy. No author names or institutional affiliations are included.

We sincerely hope this code provides the additional clarity you are looking for. We remain open to any further suggestions you may have to make this work more robust and impactful.
