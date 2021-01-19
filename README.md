# Bachelor-Thesis-Marginal-Sampling

The code which I used in this Bachelor thesis is spread over several file. I will go through them hierarchical.

\\testing

	In this folder is the python file which I used to create the Figure 1. In the subfolder \\testing\\plots you can find the plot I used in my thesis.

\\pipeline_for_sampling

	The 4 Jupiter-Notebook files pipeline_example_COnversion-Reaction_likelihood_and_prior, pipeline_example-Conversion-Reaction_marginal_likelihood,
	pipeline_example-mRNA-transfection_likelihood_and_prior and pipeline_example-mRNA-transfection_marginal_likelihood were uses to generate the sampling for the two models
	in section 3.1.4 and 3.1.5.
	All sample results were saved in the corresponding folders \\Results_CR_FP, \\Results_CR_MP, \\Results_mRNA_FP and \\Results_mRNA_MP. Because we 

\\pipeline_for_sampling\\data_convert_plot

	In this python file all plots and changes for the samplings were executed. The only exceptions are the files \\pipeline_for_sampling\\CR_offset_and_precision and 
	\\pipeline_for_sampling\\mRNA_offset_and_precision where I created the offset and precision data out of the sampled parameters for the Marginal approach.

\\pipeline_for_sampling\\plots

	Here are all plots which I created for the Gaussian noise case. They are ordered by the model and the approach for which they were created.

\\pipeline_for_sampling\\conversion_reaction

	In this folder all data which we used for the Conversion Reaction model is saved

\\pipeline_for_sampling\\mRNA-transfection

	In this folder all data which we used for the mRNA model is saved