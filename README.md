# Hybrid_Computational_Agent-Based_Tumor_Chemotherapy

Developed a hybrid computational model that integrates physiologically based pharmacokinetic model, continuum diffusion-reaction model and discrete cell automaton model to investigate invasive solid tumor growth in heterogenous microenvironment under chemotherapy.

Reference Paper: Xie H, Jiao Y, Fan Q, Hai M, Yang J, Hu Z, Yang Y, Shuai J, Chen G, Liu R, Liu L., “Modeling threedimensional invasive solid tumor growth in heterogeneous microenvironment under chemotherapy,” PLoS One., vol. 10, 2018. doi:10.1371/journal.pone.0206292. PMID: 30365511

<img width="2736" height="1765" alt="Spatial-Temporal Coupling" src="https://github.com/user-attachments/assets/f5694ddd-eb52-41ce-a653-b64b4b25002d" />

Benchmarking in High Performance Computing versus Google Cloud Platform (GCP)
<img width="4608" height="3456" alt="Poster_Vision-Lab_ECE_Walia-Farzana" src="https://github.com/user-attachments/assets/27622493-f1c5-4657-b0b1-b8affc906932" />

To run code within the HPC environment:
Example: crun -p ~/env/ssn/python/"your-directory"/hybrid_hetero_env.py

# Please cite the paper and Githib link if you apply this code for your research. The code is simulated in 2D.

Cancer growth is a multiscale, nonlinear dynamical problem where basic evolution is quantitatively explained by mathematical models. The mathematical cancer model provides an understanding of the complex biological system by unvelling the underlying mechanisms and quantitative insights. The mathematical and computational models used to simulate spatially resolved tumor growth fall into two main categories: either continuum or discrete (agent-based) with respect to the fact how tumor tissue is represented.

Continuum models treat variables, such as tumor cell density, as continuous macroscopic quantities based on conservation laws. Alternatively, discrete models represent each cell as an individual agent on the basis of a specific set of biophysical and biochemical rules. This rules are useful for investigating genetic instability, carcinogenesis, natural selection and cell-cell in addition to cell-microenvironment interactions.
